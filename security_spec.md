# Security Specification — I lang Platform (Phase 0 TDD)

## 1. Data Invariants
1. Every document under `/users/{userId}` and its subcollections (`progress`, `personalPhrases`, `testAttempts`) belongs strictly to `userId == request.auth.uid`.
2. Users must have a verified auth session (`request.auth != null && request.auth.token.email_verified == true`) for cloud writes.
3. No user can read, list, update, or delete another user's profile, email PII, progress, personal phrases, or test attempts.
4. `createdAt` and `uid`/`userId` fields are immutable after creation; `createdAt == request.time` and `updatedAt == request.time` are strictly enforced against server time.
5. `testAttempts` documents are append-only immutable records (`allow update: if false;`) to prevent post-hoc score tampering.
6. Every string and array enforces `.size()` bounds to prevent resource exhaustion ("Denial of Wallet").

## 2. The "Dirty Dozen" Attack Payloads (All Must Return PERMISSION_DENIED)
1. Unauthenticated read on `/users/u1`.
2. Cross-user read on `/users/u2` while signed in as `u1`.
3. Creating `/users/u1` with `uid: "u2"` (Identity spoofing).
4. Creating `/users/u1` with an unexpected shadow field `isAdmin: true` (Shadow field injection).
5. Creating `/users/u1` with a client-forged `createdAt` timestamp instead of `request.time`.
6. Unverified email account (`email_verified: false`) writing to `/users/u1/personalPhrases/p1`.
7. Updating `/users/u1` while trying to mutate immutable `uid` or `createdAt`.
8. Updating `/users/u1/testAttempts/a1` to inflate a failed `score` from `12` to `19`.
9. Injecting a 200KB string into `phrase` in `/users/u1/personalPhrases/p1`.
10. Injecting a malformed document ID containing slashes or >128 chars (`isValidId` guard).
11. Cross-user `list` query on `/users/u2/testAttempts` as `u1`.
12. Deleting `/users/u1/testAttempts/a1` or another user's `/users/u2/personalPhrases/p1`.
