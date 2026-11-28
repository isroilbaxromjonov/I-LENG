import 'dotenv/config';
import express, { Request, Response } from 'express';
import crypto from 'crypto';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;
const isProd = process.env.NODE_ENV === 'production';

function getAiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY is not configured on the server.');
  }
  return new GoogleGenAI({
    apiKey,
    apiVersion: 'v1beta',
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function extractReadableError(raw: string): string {
  const trimmed = raw.trim();
  if (trimmed.startsWith('{')) {
    try {
      const parsed = JSON.parse(trimmed);
      if (parsed?.error?.message) {
        return String(parsed.error.message);
      }
      if (parsed?.message) {
        return String(parsed.message);
      }
    } catch {
      // ignore invalid JSON
    }
  }
  return raw;
}

function getErrorMessage(err: unknown): string {
  const raw = err instanceof Error ? err.message : String(err);
  const msg = extractReadableError(raw);
  const lower = msg.toLowerCase();
  if (lower.includes('high demand') || lower.includes('503') || lower.includes('unavailable')) {
    return 'AI serverida vaqtincha yuklama yuqori (503). Birozdan so‘ng qayta urinib ko‘ring.';
  }
  if (lower.includes('429') || lower.includes('quota') || lower.includes('resource_exhausted')) {
    return 'So‘rovlar limiti vaqtincha to‘ldi (429). Bir necha soniyadan keyin qayta urinib ko‘ring.';
  }
  return msg;
}

function isRetryableGeminiError(err: unknown): boolean {
  const raw = (err instanceof Error ? err.message : String(err)).toLowerCase();
  const status = (err as { status?: number })?.status;
  if (status === 429 || status === 500 || status === 502 || status === 503 || status === 504) {
    return true;
  }
  return (
    raw.includes('503') ||
    raw.includes('429') ||
    raw.includes('unavailable') ||
    raw.includes('high demand') ||
    raw.includes('overloaded') ||
    raw.includes('resource_exhausted') ||
    raw.includes('econnreset') ||
    raw.includes('etimedout') ||
    raw.includes('fetch failed')
  );
}

async function generateWithRetry<T>(fn: () => Promise<T>, maxAttempts = 4): Promise<T> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      if (!isRetryableGeminiError(err) || attempt === maxAttempts) {
        throw err;
      }
      const delayMs = 450 * Math.pow(2, attempt - 1) + Math.floor(Math.random() * 180);
      console.warn(`Gemini transient error (attempt ${attempt}/${maxAttempts}), retrying in ${delayMs}ms...`);
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }
  throw lastError;
}

const app = express();
app.use(express.json({ limit: '25mb' }));

// Server-side LRU cache + in-flight coalescing for TTS audio (near-instant repeat & prefetch hits)
const MAX_TTS_CACHE_ITEMS = 250;
const ttsServerCache = new Map<string, string>();
const ttsServerInFlight = new Map<string, Promise<string>>();

function getTtsCacheKey(payload: unknown): string {
  return crypto.createHash('sha256').update(JSON.stringify(payload)).digest('hex');
}

function setTtsCacheItem(key: string, base64Audio: string): void {
  if (ttsServerCache.has(key)) {
    ttsServerCache.delete(key);
  } else if (ttsServerCache.size >= MAX_TTS_CACHE_ITEMS) {
    const oldestKey = ttsServerCache.keys().next().value;
    if (oldestKey) ttsServerCache.delete(oldestKey);
  }
  ttsServerCache.set(key, base64Audio);
}

// Health check
app.get('/api/health', (_req: Request, res: Response) => {
  res.json({ ok: true, hasApiKey: Boolean(process.env.GEMINI_API_KEY) });
});

// 1. Gemini TTS endpoint (single-speaker & dual-speaker with accent support)
app.post('/api/tts', async (req: Request, res: Response) => {
  try {
    const {
      text,
      accent = 'british',
      voiceName,
      style,
      dialogue,
    } = req.body as {
      text?: string;
      accent?: 'british' | 'american';
      voiceName?: string;
      style?: string;
      dialogue?: Array<{ speaker: string; text: string; voiceName?: string; style?: string }>;
    };

    const ai = getAiClient();
    const accentDesc =
      accent === 'american'
        ? 'Clear natural General American English accent'
        : 'Clear natural British Received Pronunciation English accent';

    // Multi-speaker dialogue using gemini-3.8-flash-tts
    if (Array.isArray(dialogue) && dialogue.length >= 2) {
      const cleanDialogue = dialogue
        .slice(0, 24)
        .filter((d) => d && typeof d.text === 'string' && d.text.trim().length > 0);
      if (cleanDialogue.length === 0) {
        return res.status(400).json({ error: 'TTS dialog matni bo‘sh.' });
      }

      const cacheKey = getTtsCacheKey({ kind: 'dialogue', accent, lines: cleanDialogue });
      const cached = ttsServerCache.get(cacheKey);
      if (cached) {
        ttsServerCache.delete(cacheKey);
        ttsServerCache.set(cacheKey, cached);
        return res.json({ audioBase64: cached, mimeType: 'audio/wav', cached: true });
      }

      let pending = ttsServerInFlight.get(cacheKey);
      if (!pending) {
        const uniqueSpeakers: Array<{ speaker: string; voiceName: string }> = [];
        const defaultVoices = ['Kore', 'Puck'];
        for (const line of cleanDialogue) {
          const cleanSpeaker = String(line.speaker || 'Speaker').trim();
          if (!uniqueSpeakers.find((s) => s.speaker === cleanSpeaker) && uniqueSpeakers.length < 2) {
            uniqueSpeakers.push({
              speaker: cleanSpeaker,
              voiceName: line.voiceName || defaultVoices[uniqueSpeakers.length] || 'Kore',
            });
          }
        }
        if (uniqueSpeakers.length === 1) {
          uniqueSpeakers.push({ speaker: 'CoSpeaker', voiceName: 'Puck' });
        }

        const parts = cleanDialogue.map((line) => {
          const spk =
            uniqueSpeakers.find((s) => s.speaker === String(line.speaker || '').trim())?.speaker ||
            uniqueSpeakers[0].speaker;
          return {
            text: `${spk}: ${String(line.text || '').slice(0, 600)}`,
            speechMetadata: {
              speaker: spk,
              style: line.style ? `${accentDesc}, ${line.style}` : `${accentDesc}, natural exam dialogue`,
            },
          };
        });

        pending = (async () => {
          const response = await generateWithRetry(() =>
            ai.models.generateContent({
              model: 'gemini-3.8-flash-tts',
              contents: [{ role: 'user', parts }],
              config: {
                responseModalities: ['AUDIO'],
                speechConfig: {
                  multiSpeakerVoiceConfig: {
                    speakerVoiceConfigs: uniqueSpeakers.map((s) => ({
                      speaker: s.speaker,
                      voiceConfig: {
                        prebuiltVoiceConfig: { voiceName: s.voiceName },
                      },
                    })),
                  },
                },
              },
            }),
          );
          const audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
          if (!audio) throw new Error('Audio javob yaratilmadi.');
          setTtsCacheItem(cacheKey, audio);
          return audio;
        })().finally(() => {
          ttsServerInFlight.delete(cacheKey);
        });
        ttsServerInFlight.set(cacheKey, pending);
      }

      const base64Audio = await pending;
      return res.json({ audioBase64: base64Audio, mimeType: 'audio/wav' });
    }

    // Single-speaker TTS using gemini-3.8-flash-lite-tts
    if (!text || typeof text !== 'string' || !text.trim()) {
      return res.status(400).json({ error: 'TTS uchun matn kiritilmadi.' });
    }

    const cleanText = text.trim().slice(0, 3500);
    const chosenVoice = voiceName || (accent === 'american' ? 'Puck' : 'Kore');
    const cacheKey = getTtsCacheKey({ kind: 'single', accent, voice: chosenVoice, style: style || '', text: cleanText });
    const cached = ttsServerCache.get(cacheKey);
    if (cached) {
      ttsServerCache.delete(cacheKey);
      ttsServerCache.set(cacheKey, cached);
      return res.json({ audioBase64: cached, mimeType: 'audio/wav', cached: true });
    }

    let pending = ttsServerInFlight.get(cacheKey);
    if (!pending) {
      pending = (async () => {
        const response = await generateWithRetry(() =>
          ai.models.generateContent({
            model: 'gemini-3.8-flash-lite-tts',
            contents: [
              {
                role: 'user',
                parts: [
                  {
                    text: cleanText,
                    speechMetadata: {
                      style: style ? `${accentDesc}, ${style}` : `${accentDesc}, articulate language examiner`,
                    },
                  },
                ],
              },
            ],
            config: {
              responseModalities: ['AUDIO'],
              speechConfig: {
                voiceConfig: {
                  prebuiltVoiceConfig: { voiceName: chosenVoice },
                },
              },
            },
          }),
        );
        const audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
        if (!audio) throw new Error('Audio oqimi bo‘sh qaytdi.');
        setTtsCacheItem(cacheKey, audio);
        return audio;
      })().finally(() => {
        ttsServerInFlight.delete(cacheKey);
      });
      ttsServerInFlight.set(cacheKey, pending);
    }

    const base64Audio = await pending;
    return res.json({ audioBase64: base64Audio, mimeType: 'audio/wav' });
  } catch (err) {
    console.error('TTS error:', err);
    return res.status(500).json({ error: getErrorMessage(err) });
  }
});

// 2. Audio transcription endpoint using gemini-3.5-transcribe
app.post('/api/transcribe', async (req: Request, res: Response) => {
  try {
    const { audioBase64, mimeType = 'audio/webm' } = req.body as {
      audioBase64?: string;
      mimeType?: string;
    };

    if (!audioBase64 || typeof audioBase64 !== 'string' || audioBase64.length < 40) {
      return res.status(400).json({ error: 'Audio ma’lumoti juda qisqa yoki yuborilmadi.' });
    }

    const mimeMap: Record<string, string> = {
      'audio/webm': 'audio/webm',
      'audio/wav': 'audio/wav',
      'audio/wave': 'audio/wav',
      'audio/x-wav': 'audio/wav',
      'audio/mp3': 'audio/mp3',
      'audio/mpeg': 'audio/mp3',
      'audio/ogg': 'audio/ogg',
      'audio/mp4': 'audio/mp4',
      'audio/x-m4a': 'audio/mp4',
      'audio/aac': 'audio/aac',
      'audio/flac': 'audio/flac',
    };
    const baseMime = String(mimeType || 'audio/webm').split(';')[0].trim().toLowerCase();
    const safeMime = mimeMap[baseMime] || 'audio/webm';

    const ai = getAiClient();
    const response = await generateWithRetry(() =>
      ai.models.generateContent({
        model: 'gemini-3.5-transcribe',
        contents: {
          parts: [
            {
              inlineData: {
                mimeType: safeMime,
                data: audioBase64,
              },
            },
            {
              text: 'Transcribe the candidate\'s English speech verbatim, preserving natural phrasing or grammar mistakes accurately as spoken. If the audio is silent or contains only noise, return an empty string.',
            },
          ],
        },
      }),
    );

    return res.json({ transcript: (response.text ?? '').trim() });
  } catch (err) {
    console.error('Transcription error:', err);
    return res.status(500).json({ error: getErrorMessage(err) });
  }
});

// 3. Writing evaluation endpoint (IELTS, Multilevel, CEFR separated)
app.post('/api/evaluate-writing', async (req: Request, res: Response) => {
  try {
    const { track, cefrLevel, taskType, prompt, essay, targetWordCount } = req.body as {
      track: 'ielts' | 'multilevel' | 'cefr';
      cefrLevel?: string;
      taskType: string;
      prompt: string;
      essay: string;
      targetWordCount?: number;
    };

    if (!essay || typeof essay !== 'string' || essay.trim().length < 15) {
      return res.status(400).json({ error: 'Iltimos, baholash uchun kamida bir nechta gap yozing.' });
    }

    const ai = getAiClient();
    const rubricContext =
      track === 'ielts'
        ? `Evaluate strictly according to official IELTS Writing Band Descriptors (0.0 to 9.0 in 0.5 steps). Criteria: Task Achievement / Task Response, Coherence and Cohesion, Lexical Resource, Grammatical Range and Accuracy. Do NOT inflate scores. Minimum expected words: ${targetWordCount || 250}.`
        : track === 'multilevel'
          ? `Evaluate strictly according to Uzbekistan National Certificate (BMBA Multilevel) Writing Rubric (score out of 75 scale and CEFR band B1/B2/C1). Criteria: Task Fulfilment & Register, Organization & Cohesion, Vocabulary Range & Precision, Grammatical Accuracy. Minimum expected words: ${targetWordCount || 150}.`
          : `Evaluate strictly against CEFR ${cefrLevel || 'B1'} Writing descriptors (score out of 100 and achieved CEFR band). Criteria: Communicative Achievement, Text Organization, Vocabulary Control, Grammatical Accuracy.`;

    const response = await generateWithRetry(() =>
      ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: `Track: ${track.toUpperCase()} (${cefrLevel || 'Standard'})
Task Type: ${taskType}
Prompt: ${prompt}
Student Submission:
"""
${essay.slice(0, 8000)}
"""`,
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          systemInstruction: `You are a strict, honest, veteran English exam assessor. ${rubricContext}
Score each of the 4 criteria on BOTH the official scale AND a 0-5 point training scale (scoreOutOf5, integer or 0.5 step) so their sum equals scoreOutOf20 (0 to 20 max).
Include a concise disclaimerUz stating: "Bu AI bahosi rasmiy imtihon natijasining taxminiy o'quv bahosi hisoblanadi."
Provide all explanations, error analysis, and study recommendations in clear Uzbek (with English quotes/examples where helpful). Never flatter empty or short essays; apply authentic penalty for under-length or off-topic writing.`,
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              overallScore: { type: Type.STRING, description: 'e.g. "6.5" for IELTS, "58/75 (B2)" for Multilevel, or "78/100 (B1+)" for CEFR' },
              numericScore: { type: Type.NUMBER, description: 'Numeric value (0-9 for IELTS, 0-75 for Multilevel, 0-100 for CEFR)' },
              scoreOutOf20: { type: Type.NUMBER, description: 'Sum of 4 criteria out of 20 (each 0-5)' },
              disclaimerUz: { type: Type.STRING },
              wordCount: { type: Type.INTEGER },
              criteria: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    name: { type: Type.STRING },
                    score: { type: Type.STRING },
                    scoreOutOf5: { type: Type.NUMBER },
                    feedback: { type: Type.STRING },
                  },
                  required: ['name', 'score', 'scoreOutOf5', 'feedback'],
                },
              },
              errors: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    original: { type: Type.STRING },
                    correction: { type: Type.STRING },
                    explanation: { type: Type.STRING },
                  },
                  required: ['original', 'correction', 'explanation'],
                },
              },
              modelAnswer: { type: Type.STRING, description: 'High-scoring model answer in English' },
              usefulVocabulary: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    word: { type: Type.STRING },
                    meaningUz: { type: Type.STRING },
                    example: { type: Type.STRING },
                  },
                  required: ['word', 'meaningUz', 'example'],
                },
              },
              summaryRecommendation: { type: Type.STRING },
            },
            required: [
              'overallScore',
              'numericScore',
              'scoreOutOf20',
              'disclaimerUz',
              'wordCount',
              'criteria',
              'errors',
              'modelAnswer',
              'usefulVocabulary',
              'summaryRecommendation',
            ],
          },
        },
      }),
    );

    const parsed = JSON.parse((response.text ?? '{}').trim());
    return res.json(parsed);
  } catch (err) {
    console.error('Writing eval error:', err);
    return res.status(500).json({ error: getErrorMessage(err) });
  }
});

// 4. AI Speaking Tutor live conversation turn (with immediate per-answer criterion scoring & better alternative)
app.post('/api/speaking-turn', async (req: Request, res: Response) => {
  try {
    const { track, cefrLevel, part, topic, currentQuestion, history, latestUserResponse } = req.body as {
      track: 'ielts' | 'multilevel' | 'cefr';
      cefrLevel?: string;
      part: string;
      topic: string;
      currentQuestion?: string;
      history: Array<{ role: 'examiner' | 'candidate'; text: string }>;
      latestUserResponse: string;
    };

    if (!latestUserResponse || typeof latestUserResponse !== 'string' || !latestUserResponse.trim()) {
      return res.status(400).json({ error: 'Javob matni bo‘sh.' });
    }

    const ai = getAiClient();
    const transcriptLog = (history || [])
      .slice(-8)
      .map((h) => `${h.role === 'examiner' ? 'Examiner' : 'Candidate'}: ${h.text}`)
      .join('\n');

    const trackRubric =
      track === 'ielts'
        ? `IELTS Speaking criteria (Fluency & Coherence, Lexical Resource, Grammar Range & Accuracy, Clarity) — score THIS single answer honestly on the IELTS Band scale (e.g. "Band 5.5", "Band 6.5", "Band 7.5").`
        : track === 'multilevel'
          ? `Uzbekistan Milliy Sertifikat (Multilevel) Speaking rubric (0–75 scale & CEFR band) — score THIS single answer honestly (e.g. "46/75 (B1)", "58/75 (B2)", "68/75 (C1)").`
          : `CEFR ${cefrLevel || 'B1'} Speaking criteria — score THIS single answer honestly (e.g. "A2+", "B1 (70/100)", "B2 (84/100)").`;

    const response = await generateWithRetry(() =>
      ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: `Exam Track: ${track.toUpperCase()} (${cefrLevel || 'Standard'})
Section: ${part}
Topic: ${topic}
Question Asked: "${currentQuestion || 'Current prompt'}"
Recent Dialogue:
${transcriptLog}
Candidate's Latest Answer: "${latestUserResponse.slice(0, 2500)}"

Tasks for the AI Speaking Tutor:
1. Analyze ONLY the candidate's latest answer against the official ${track.toUpperCase()} rubric and assign an honest score (turnScore).
2. In analysisUz (clear Uzbek, 2-3 sentences), explain WHY this answer earned that score (grammar accuracy, vocabulary level, coherence/development, or missing details). Never flatter weak/short answers.
3. In betterAlternativeEn (natural spoken English), write a DIFFERENT, significantly improved high-scoring answer (Band 8.0+ / C1 level) to the same question using richer collocations and natural linking phrases.
4. In examinerReply (1-2 natural spoken English sentences), briefly acknowledge the candidate and ask the next relevant follow-up question for ${part}.`,
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          systemInstruction: `You are an expert ${track.toUpperCase()} Speaking Coach and Examiner. ${trackRubric} Be honest, constructive, and specific.`,
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              turnScore: { type: Type.STRING, description: 'Criterion score for this single response, e.g. "Band 6.0" or "54/75 (B2)"' },
              analysisUz: { type: Type.STRING, description: 'Detailed 2-3 sentence Uzbek evaluation of grammar, vocabulary, and relevance' },
              betterAlternativeEn: { type: Type.STRING, description: 'A different, Band 8+ / C1 phrasing of the answer in natural spoken English' },
              examinerReply: { type: Type.STRING, description: 'Natural spoken English acknowledgement + next follow-up question' },
            },
            required: ['turnScore', 'analysisUz', 'betterAlternativeEn', 'examinerReply'],
          },
        },
      }),
    );

    const parsed = JSON.parse((response.text ?? '{}').trim());
    return res.json(parsed);
  } catch (err) {
    console.error('Speaking turn error:', err);
    return res.status(500).json({ error: getErrorMessage(err) });
  }
});

// 5. Full Speaking Evaluation endpoint
app.post('/api/evaluate-speaking', async (req: Request, res: Response) => {
  try {
    const { track, cefrLevel, part, topic, exchanges } = req.body as {
      track: 'ielts' | 'multilevel' | 'cefr';
      cefrLevel?: string;
      part: string;
      topic: string;
      exchanges: Array<{ question: string; answer: string }>;
    };

    if (!Array.isArray(exchanges) || exchanges.length === 0) {
      return res.status(400).json({ error: 'Baholash uchun kamida bitta javob bo‘lishi kerak.' });
    }

    const ai = getAiClient();
    const formatted = exchanges
      .map((ex, i) => `Q${i + 1}: ${ex.question}\nCandidate Answer: ${ex.answer}`)
      .join('\n\n');

    const rubric =
      track === 'ielts'
        ? `Apply official IELTS Speaking criteria:
1. Fluency and Coherence
2. Lexical Resource
3. Grammatical Range and Accuracy
4. Pronunciation (estimate clarity, phrasing, stress/intonation indicators from transcript structure and word choices; state clearly what was inferred from transcript).
Score each 0.0-9.0 in 0.5 steps and compute overall estimated IELTS Speaking band.`
        : track === 'multilevel'
          ? `Apply Uzbekistan Milliy Sertifikat (Multilevel) Speaking rubric (0-75 scale & CEFR B1/B2/C1). Criteria: Fluency & Coherence, Lexical Resource, Grammatical Accuracy, Pronunciation & Intelligibility.`
          : `Apply CEFR ${cefrLevel || 'B1'} Speaking descriptors (0-100 scale & CEFR level). Criteria: Range, Accuracy, Fluency, Interaction & Coherence.`;

    const response = await generateWithRetry(() =>
      ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: `Track: ${track.toUpperCase()} (${cefrLevel || 'Standard'})
Section: ${part}
Topic: ${topic}
Interview Transcript:
${formatted}`,
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          systemInstruction: `You are an uncompromising, honest English Speaking Examiner. ${rubric}
Score each of the 4 criteria on BOTH the official scale AND a 0-5 point training scale (scoreOutOf5) so their sum equals scoreOutOf20 (0 to 20 max).
Include disclaimerUz: "Bu AI bahosi taxminiy o'quv bahosi bo'lib, rasmiy imtihon komissiyasi qarorini almashtirmaydi."
Do NOT simply praise the candidate. Identify real grammatical mistakes, weak vocabulary, repetition, or underdeveloped ideas. Provide error explanations, correction techniques, improved sample answers, and next-practice recommendations in clear Uzbek (with English examples).`,
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              overallScore: { type: Type.STRING },
              numericScore: { type: Type.NUMBER },
              scoreOutOf20: { type: Type.NUMBER },
              disclaimerUz: { type: Type.STRING },
              criteria: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    name: { type: Type.STRING },
                    score: { type: Type.STRING },
                    scoreOutOf5: { type: Type.NUMBER },
                    analysis: { type: Type.STRING },
                  },
                  required: ['name', 'score', 'scoreOutOf5', 'analysis'],
                },
              },
              errors: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    said: { type: Type.STRING },
                    fix: { type: Type.STRING },
                    howToCorrect: { type: Type.STRING },
                  },
                  required: ['said', 'fix', 'howToCorrect'],
                },
              },
              improvedSampleAnswer: { type: Type.STRING },
              nextPracticeAdvice: { type: Type.STRING },
            },
            required: [
              'overallScore',
              'numericScore',
              'scoreOutOf20',
              'disclaimerUz',
              'criteria',
              'errors',
              'improvedSampleAnswer',
              'nextPracticeAdvice',
            ],
          },
        },
      }),
    );

    const parsed = JSON.parse((response.text ?? '{}').trim());
    return res.json(parsed);
  } catch (err) {
    console.error('Speaking eval error:', err);
    return res.status(500).json({ error: getErrorMessage(err) });
  }
});

// 6. Explain mistake / lesson concept via Gemini AI
app.post('/api/explain', async (req: Request, res: Response) => {
  try {
    const { track, contextText, question, userAnswer, correctAnswer, topic } = req.body as {
      track: string;
      contextText?: string;
      question?: string;
      userAnswer?: string;
      correctAnswer?: string;
      topic?: string;
    };

    const ai = getAiClient();
    const promptText = topic
      ? `Explain this ${track.toUpperCase()} English topic clearly in Uzbek with 2 crisp English examples: ${topic}\nContext: ${contextText || ''}`
      : `Explain in Uzbek why the student's answer is wrong (or how to solve this ${track.toUpperCase()} question).
Passage/Script excerpt: ${(contextText || '').slice(0, 2500)}
Question: ${question || ''}
Student Answer: ${userAnswer || '(bo‘sh)'}
Correct Answer: ${correctAnswer || ''}`;

    const response = await generateWithRetry(() =>
      ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: promptText,
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          systemInstruction:
            'You are a precise Uzbek-speaking English exam coach. Be concise, educational, and quote the exact evidence from the passage or script.',
        },
      }),
    );

    return res.json({ explanation: (response.text ?? '').trim() });
  } catch (err) {
    console.error('Explain error:', err);
    return res.status(500).json({ error: getErrorMessage(err) });
  }
});

// 7. Dynamic fresh exam material generator (supports format updates & non-repeating practice)
app.post('/api/generate-material', async (req: Request, res: Response) => {
  try {
    const { track, cefrLevel, skill, subType } = req.body as {
      track: 'ielts' | 'multilevel' | 'cefr';
      cefrLevel?: string;
      skill: 'listening' | 'reading' | 'writing' | 'speaking';
      subType?: string;
    };

    const ai = getAiClient();

    if (skill === 'reading') {
      const response = await generateWithRetry(() =>
        ai.models.generateContent({
          model: 'gemini-4-hydrogen',
          contents: `Generate a fresh, realistic ${track.toUpperCase()} ${cefrLevel ? `(${cefrLevel})` : ''} Reading exercise focusing on ${subType || 'mixed official tasks'}.
Include 3-4 labeled paragraphs (A, B, C, D) and 4 high-quality exam questions (mix of True/False/Not Given, Multiple Choice, and Sentence Completion or Short Answer) with exact passage evidence and Uzbek explanation.`,
          config: {
            thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                title: { type: Type.STRING },
                levelLabel: { type: Type.STRING },
                taskTypeLabel: { type: Type.STRING },
                paragraphs: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      label: { type: Type.STRING },
                      text: { type: Type.STRING },
                    },
                    required: ['label', 'text'],
                  },
                },
                questions: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      id: { type: Type.STRING },
                      type: { type: Type.STRING },
                      prompt: { type: Type.STRING },
                      options: { type: Type.ARRAY, items: { type: Type.STRING } },
                      answer: { type: Type.STRING },
                      evidenceQuote: { type: Type.STRING },
                      explanationUz: { type: Type.STRING },
                    },
                    required: ['id', 'type', 'prompt', 'answer', 'evidenceQuote', 'explanationUz'],
                  },
                },
              },
              required: ['title', 'levelLabel', 'taskTypeLabel', 'paragraphs', 'questions'],
            },
          },
        }),
      );
      return res.json(JSON.parse((response.text ?? '{}').trim()));
    }

    if (skill === 'listening') {
      const response = await generateWithRetry(() =>
        ai.models.generateContent({
          model: 'gemini-4-hydrogen',
          contents: `Generate a fresh ${track.toUpperCase()} ${cefrLevel ? `(${cefrLevel})` : ''} Listening exercise (${subType || 'Dialogue & Note Completion'}).
Provide a realistic 6-8 line spoken script with two speakers (Alex and Sarah) or single lecturer, plus 4 exam questions (Multiple Choice, Form/Note Completion, Sentence Completion) with answers and Uzbek explanations.`,
          config: {
            thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                title: { type: Type.STRING },
                stageLabel: { type: Type.STRING },
                taskTypeLabel: { type: Type.STRING },
                dialogue: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      speaker: { type: Type.STRING },
                      text: { type: Type.STRING },
                    },
                    required: ['speaker', 'text'],
                  },
                },
                questions: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      id: { type: Type.STRING },
                      type: { type: Type.STRING },
                      prompt: { type: Type.STRING },
                      options: { type: Type.ARRAY, items: { type: Type.STRING } },
                      answer: { type: Type.STRING },
                      explanationUz: { type: Type.STRING },
                    },
                    required: ['id', 'type', 'prompt', 'answer', 'explanationUz'],
                  },
                },
              },
              required: ['title', 'stageLabel', 'taskTypeLabel', 'dialogue', 'questions'],
            },
          },
        }),
      );
      return res.json(JSON.parse((response.text ?? '{}').trim()));
    }

    if (skill === 'writing') {
      const response = await generateWithRetry(() =>
        ai.models.generateContent({
          model: 'gemini-4-hydrogen',
          contents: `Generate a fresh ${track.toUpperCase()} ${cefrLevel ? `(${cefrLevel})` : ''} Writing prompt for task type: ${subType || 'Essay'}. Include clear exam instructions, minimum word count, and 5 high-value academic vocabulary items with Uzbek meanings.`,
          config: {
            thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                title: { type: Type.STRING },
                taskType: { type: Type.STRING },
                prompt: { type: Type.STRING },
                minWords: { type: Type.INTEGER },
                timeMinutes: { type: Type.INTEGER },
                tipsUz: { type: Type.STRING },
              },
              required: ['title', 'taskType', 'prompt', 'minWords', 'timeMinutes', 'tipsUz'],
            },
          },
        }),
      );
      return res.json(JSON.parse((response.text ?? '{}').trim()));
    }

    // Speaking fresh prompt
    const response = await generateWithRetry(() =>
      ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: `Generate a fresh ${track.toUpperCase()} ${cefrLevel ? `(${cefrLevel})` : ''} Speaking prompt for ${subType || 'Part 2 & Part 3'}. Include 3 Part 1 warm-up questions, 1 Cue Card with 4 bullet points, and 3 Part 3 analytical questions.`,
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              topic: { type: Type.STRING },
              part1Questions: { type: Type.ARRAY, items: { type: Type.STRING } },
              cueCardPrompt: { type: Type.STRING },
              cueCardBullets: { type: Type.ARRAY, items: { type: Type.STRING } },
              part3Questions: { type: Type.ARRAY, items: { type: Type.STRING } },
            },
            required: ['topic', 'part1Questions', 'cueCardPrompt', 'cueCardBullets', 'part3Questions'],
          },
        },
      }),
    );
    return res.json(JSON.parse((response.text ?? '{}').trim()));
  } catch (err) {
    console.error('Generate material error:', err);
    return res.status(500).json({ error: getErrorMessage(err) });
  }
});

// 8. Lexicon batch generator (3,000 Words & 500 Phrases expansion per exam)
app.post('/api/lexicon-batch', async (req: Request, res: Response) => {
  try {
    const {
      track,
      cefrLevel,
      kind,
      topicOrCategory,
      excludeList = [],
    } = req.body as {
      track: 'ielts' | 'multilevel' | 'cefr' | 'foundation';
      cefrLevel?: string;
      kind: 'words' | 'phrases';
      topicOrCategory: string;
      excludeList?: string[];
    };

    const ai = getAiClient();
    const excludePreview = excludeList.slice(-80).join(', ');

    if (kind === 'words') {
      const response = await generateWithRetry(() =>
        ai.models.generateContent({
          model: 'gemini-4-hydrogen',
          contents: `Generate 20 essential, high-utility English vocabulary words for the ${track.toUpperCase()} ${cefrLevel ? `(${cefrLevel})` : ''} curriculum under the topic "${topicOrCategory}".
Do NOT repeat any of these existing words: [${excludePreview}].
Each word must include phonetic transcription, partOfSpeech, accurate Uzbek meaning, one natural exam-appropriate English example sentence + Uzbek translation (exampleUz), a common collocation, 2-3 single-word English synonyms, and 1-2 English antonyms.`,
          config: {
            thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  word: { type: Type.STRING },
                  phonetic: { type: Type.STRING },
                  partOfSpeech: { type: Type.STRING },
                  meaningUz: { type: Type.STRING },
                  exampleEn: { type: Type.STRING },
                  exampleUz: { type: Type.STRING },
                  collocation: { type: Type.STRING },
                  synonyms: { type: Type.ARRAY, items: { type: Type.STRING } },
                  antonyms: { type: Type.ARRAY, items: { type: Type.STRING } },
                },
                required: ['word', 'phonetic', 'partOfSpeech', 'meaningUz', 'exampleEn', 'exampleUz', 'collocation', 'synonyms', 'antonyms'],
              },
            },
          },
        }),
      );
      return res.json({ items: JSON.parse((response.text ?? '[]').trim()) });
    }

    const response = await generateWithRetry(() =>
      ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: `Generate 15 high-scoring, practical English phrases/expressions for ${track.toUpperCase()} ${cefrLevel ? `(${cefrLevel})` : ''} in the functional category "${topicOrCategory}".
Do NOT repeat any of these existing phrases: [${excludePreview}].
Provide accurate Uzbek translation, usageSituationUz (when to use it in Uzbek), 1 natural English example sentence + Uzbek translation (exampleUz), register ("Speaking", "Writing", or "Both"), 2-3 alternative/replacement English phrases, and skillBenefit (1 short Uzbek sentence on how it boosts Speaking/Writing score).`,
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                phrase: { type: Type.STRING },
                meaningUz: { type: Type.STRING },
                usageSituationUz: { type: Type.STRING },
                exampleEn: { type: Type.STRING },
                exampleUz: { type: Type.STRING },
                register: { type: Type.STRING },
                alternatives: { type: Type.ARRAY, items: { type: Type.STRING } },
                skillBenefit: { type: Type.STRING },
              },
              required: ['phrase', 'meaningUz', 'usageSituationUz', 'exampleEn', 'exampleUz', 'register', 'alternatives', 'skillBenefit'],
            },
          },
        },
      }),
    );
    return res.json({ items: JSON.parse((response.text ?? '[]').trim()) });
  } catch (err) {
    console.error('Lexicon batch error:', err);
    return res.status(500).json({ error: getErrorMessage(err) });
  }
});

// 9. Global Top-Bar AI Assistant (Knows everything about the I lang app + translates words/phrases)
app.post('/api/app-assistant', async (req: Request, res: Response) => {
  try {
    const { messages = [], appContext } = req.body as {
      messages?: Array<{ role: 'user' | 'assistant'; text: string }>;
      appContext?: {
        activeTab?: string;
        programKind?: string;
        currentStage?: string;
        selectedTrack?: string;
        cefrLevel?: string;
      };
    };

    const validMsgs = (Array.isArray(messages) ? messages : [])
      .filter((m) => m && typeof m.text === 'string' && m.text.trim())
      .slice(-12);

    if (validMsgs.length === 0) {
      return res.status(400).json({ error: 'Savol matni bo‘sh bo‘lmasligi kerak.' });
    }

    const ai = getAiClient();
    const conversationText = validMsgs
      .map((m) => `${m.role === 'user' ? 'Foydalanuvchi' : 'I lang Yordamchi'}: ${m.text.trim()}`)
      .join('\n');

    const contextSummary = appContext
      ? `Hozirgi holat: bo‘lim="${appContext.activeTab || 'home'}", dastur="${appContext.programKind || 'foundation'}", Learn bosqichi="${appContext.currentStage || 'beginner-1'}", tanlangan imtihon="${appContext.selectedTrack || 'ielts'}"${appContext.cefrLevel ? ` (${appContext.cefrLevel})` : ''}.`
      : '';

    const systemInstruction = `Siz "I lang" ingliz tili o‘rganish va imtihonlarga tayyorgarlik platformasining rasmiy AI yordamchisisiz.
Asosiy vazifalaringiz:
1. Foydalanuvchiga "I lang" ilovasining barcha bo‘limlari va funksiyalarini tushuntirish va yo‘l ko‘rsatish.
2. Inglizcha ↔ O‘zbekcha so‘zlar, frazalar va gaplarni aniq tarjima qilish (transkripsiya/talaffuz eslatmasi, so‘z turkumi, 1–2 ta tabiiy misol gapi va sinonimlar bilan).
3. Grammatika, imtihon mezonlari va samarali tayyorgarlik bo‘yicha qisqa, amaliy maslahat berish.

"I lang" ILOVASI HAQIDA TO‘LIQ MA’LUMOT:
- Boshlang‘ich ekran (Onboarding): 2 ta yo‘nalish tanlanadi: (1) "Ingliz tilini noldan o‘rganish" (Beginner, Elementary, Intermediate), (2) "Sertifikat imtihonlariga tayyorgarlik" (IELTS, CEFR A1–C1, Multilevel).
- Asosiy 6 ta bo‘lim:
  1) Home: Kunlik maqsadlar, davom ettirilayotgan dars, umumiy natijalar va tezkor kirish kartalari.
  2) Learn: 9 ta bosqich (Beginner 1-3, Elementary 1-3, Pre-Intermediate, Intermediate, Upper-Intermediate). Har bir bosqich ichida Grammar, Listening, Reading, Writing, Speaking va Yakuniy test (Final Test — keyingi bosqichga o‘tish uchun kamida 17/20 ball kerak) mavjud.
  3) Exams: 3 ta mustaqil sertifikat imtihoni:
     - IELTS: Academic/General, 0.0–9.0 Band (0.5 qadam), Listening, Reading, Writing (Task 1 150+ so‘z + Task 2 250+ so‘z), Speaking (Parts 1, 2, 3).
     - CEFR: A1, A2, B1, B2, C1 alohida darajalar, 0–100% mezon.
     - Multilevel (O‘zbekiston BMBA Milliy Sertifikat): 0–75 ball (B1: 35+, B2: 51+, C1: 65+), Writing (Task 1 xat + Task 2 insho), Speaking (Part 1.1, Part 1.2 Rasm tasvirlash, Part 2 Monolog, Part 3 Munozara).
     - Har bir imtihonda Practice Tests va rasmiy nizom bo‘yicha Full Mock bor. Full Mock da Listening → Reading → Writing → Speaking ketma-ket olinadi. Speaking qismida foydalanuvchi imtihon xonasida AI ustoz qarshisida o‘tiradi: ustoz savolni ham yozma, ham ovozli beradi; talaba javobni FAQAT ovozli (mikrofon orqali) beradi; javoblar eslab qolinib yakunda barcha 4 ta skill mezon bo‘yicha baholanadi.
  4) Vocabulary:
     - So‘zlar A–Z: Alifbo tartibida so‘zlar, tarjima, misol, birikma, antonimlar va bosiladigan sinonimlar (sinonim ustiga bosilsa o‘sha so‘z kartasiga o‘tadi), Flashcard mashqi va "+20 ta yangi so‘z" qo‘shish.
     - Frazalar: 8 ta tayyor bo‘lim (Kirish, Chiqish/Yakunlash, Ikkita gapni ulash, Boshqa fikrga o‘tish, Misol keltirish, Taqqoslash, Xulosa, Vaqtni cho‘zish) — har birida 40 tadan tayyor fraza, ma’nosi, ishlatilish vaziyati, misoli va muqobillari bor. Shaxsiy fraza kiritish esa faqat "O‘zimning frazalarim" bo‘limida (50 tagacha) mavjud.
  5) Progress: Kunlik/haftalik faollik, skill ballari, zaif mavzular va test tarixi.
  6) Profile: Ixtiyoriy Google hisobi yoki Email/parol bilan kirish (ma’lumotlarni bulutda saqlash uchun), aksent (British/American), Light/Dark rejim, kunlik maqsadlar.
- Yuqori panelda Telefon / Planshet / Kompyuter ko‘rinishiga o‘zgartirish tugmalari ham mavjud.

Javob uslubi: O‘zbek tilida tushunarli, do‘stona, aniq va lo‘nda javob bering. Agar so‘z yoki fraza tarjimasi so‘ralsa, tarjima + talaffuz/turkum + misol gap + sinonim/maslahat ko‘rinishida chiroyli tartiblab bering.`;

    const response = await generateWithRetry(() =>
      ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: `${contextSummary}\n\nSuhbat:\n${conversationText}\n\nOxirgi foydalanuvchi xabariga aniq va foydali javob bering:`,
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          systemInstruction,
        },
      }),
    );

    const reply = (response.text ?? '').trim();
    if (!reply) {
      return res.status(502).json({ error: 'Yordamchi javob qaytara olmadi. Qayta urinib ko‘ring.' });
    }
    return res.json({ reply });
  } catch (err) {
    console.error('App assistant error:', err);
    return res.status(500).json({ error: getErrorMessage(err) });
  }
});

async function startServer() {
  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
