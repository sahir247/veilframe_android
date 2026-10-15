# VeilFrame QR Engineering Agent Instructions

## Scope

These instructions apply to the VeilFrame Android QR subsystem, especially:
- `android/app/src/main/java/com/veilframe/app/qr/`
- `android/app/src/test/java/com/veilframe/app/qr/`

The EF parity reference is EFQRCode 7.0.3 at the repository's pinned reference commit. You have full access to the local EFQRCode sources in `../EFQRCode`.

Do not rely on previous agent summaries when source inspection can verify the claim directly.

## 1. Engineering Principles

### Verify before modifying
Before editing:
1. Inspect the production implementation.
2. Inspect callers.
3. Inspect existing tests.
4. Inspect EFQRCode source for parity claims.
5. Identify whether the issue is model conversion, geometry, rendering, encoding, propagation, testing, or verification.

### Do not confuse architecture with output parity
A shared IR proves that two backends consume the same geometry definition. It does not prove identical pixels, gradients, alpha compositing, anti-aliasing, SVG rasterization, or decoding.

### Never claim completion merely because tests pass
Use these evidence levels:
- SOURCE VERIFIED
- CODE VERIFIED
- UNIT TEST VERIFIED
- RASTER VERIFIED
- DECODE VERIFIED
- NOT VERIFIED

Never claim exact/pixel parity unless the required verification was actually performed.

### Preserve intentional VeilFrame extensions
Do not remove VeilFrame-only features merely to resemble EFQRCode. Document intentional divergences.

## 2. Repository Exploration

For non-trivial QR work, inspect at least:
- `QrGenerator.kt`
- `QrDesign.kt`
- `QrGeometryIr.kt`
- `BasicGeometryBuilder.kt`
- `IrCanvasRenderer.kt`
- `IrSvgRenderer.kt`
- `AnimatedQrGenerator.kt`
- `BasicRenderer.kt`
- `IrBackedQrRenderer.kt`
- `CrossBackendCanvasSvgParityTest.kt`

Search before changing:
- `fromQrStyleParams`
- `generateAnimatedSvg`
- `BackgroundStyle.LinearGradient`
- `GeometryFill.LinearGradient`
- `angleDegrees`
- `generateBitmap(`
- `generateBitmapResult`
- `BasicGeometryProfile.EF_PARITY`

## 3. EFQRCode Parity

EF behavior must be source-derived. Verify, where relevant:
- BASIC defaults
- finder geometry
- timing/alignment geometry
- data geometry
- rounded radius
- quiet zone
- backdrop/SVG coordinates
- node order
- gradients
- alpha
- logo order
- animation

Do not infer EF behavior from names alone.

### Explicit shape mapping

At the `QrStyleParams -> QrDesign` boundary:
- `RECTANGLE -> SQUARE`
- `ROUND -> CIRCLE`
- `ROUNDED_RECTANGLE -> ROUNDED`

Unknown values should have a deterministic safe fallback unless the model explicitly requires another mapping.

Default BASIC timing/alignment must remain square.

Add regression tests that directly inspect the resulting `QrDesign`.

## 4. GenerationMode and Effective Design

There must be one authoritative parity-normalization helper, such as:

```kotlin
effectiveDesignForMode(design, mode)
```

For `PARITY_EF`:
- set `basicProfile = BasicGeometryProfile.EF_PARITY`
- default quiet zone to 1 only when no explicit, directional, or fractional quiet-zone override exists
- preserve explicit quiet-zone intent
- preserve unrelated design fields

For `SAFE` and `ARTISTIC_ENGINE`, do not mutate unrelated design state.

Use the same effective design for:
- `generateSvg(content, ...)`
- `generateWithResult(...)`
- `generateAnimatedSvg(...)`
- other high-level mode-aware generation paths

Do not duplicate parity normalization logic.

## 5. Animated SVG

Animated SVG must use the same normalized design as static SVG.

Correct flow:

```text
design + mode
    -> effectiveDesignForMode()
    -> generateMatrix()
    -> AnimatedQrGenerator
    -> SVG
```

Do not generate a PARITY_EF matrix and then render it using the original VEILFRAME design.

Add a regression test proving animated PARITY_EF uses EF parity design and preserves explicit quiet-zone customization.

## 6. Canonical Geometry IR

For BASIC:

```text
QrMatrix + QrDesign + QrGeometry
              |
      BasicGeometryBuilder
              |
        QrGeometryIr
          /             Canvas       SVG
```

The IR is the mathematical source of truth for shared BASIC geometry.

It should preserve:
- coordinates
- dimensions
- fills
- gradients
- strokes
- opacity
- paths
- transforms
- images
- masks/clipping
- ordering

Do not independently recompute equivalent geometry in Canvas and SVG when the IR can represent it.

## 7. Gradient Rules

`GeometryFill.LinearGradient` is the canonical cross-backend representation.

If `BackgroundStyle.LinearGradient` has `angleDegrees`, that value must actually affect production rendering.

Adding a helper such as `GeometryFill.LinearGradient.fromAngle()` is not enough if the production path still uses a fixed diagonal.

Required flow:

```text
BackgroundStyle.LinearGradient(angle)
          -> canonical GeometryFill
          -> Canvas + SVG
```

Canvas and SVG must consume the same endpoints from the IR.

Test at least:
- 0°
- 45°
- 90°
- 180°

Use non-square dimensions so accidental fixed-diagonal behavior cannot pass.

## 8. Canvas vs SVG Verification

Keep structural IR tests.

Also add true differential testing where technically possible:
- same output dimensions
- same background
- same geometry
- same gradient
- same alpha/colors
- same quiet zone

Measure:
- total pixels
- differing pixels
- maximum channel delta
- mean channel delta
- percentage differing

Do not fake a pixel test by comparing SVG strings to Canvas draw calls.

If reliable JVM SVG rasterization is impossible, document the limitation and use the strongest feasible integration/instrumentation test instead.

## 9. Bitmap Failure Handling

Prefer:

```text
generateBitmapResult()
    -> Success(bitmap)
    -> Failure(QrError.Rendering...)
```

Production code must not silently convert allocation/render failures into `null -> skip -> continue`.

Search all callers of deprecated nullable bitmap APIs and migrate internal production callers where possible.

## 10. Quiet Zones

Matrix size is required for quiet-zone resolution.

Prefer:
- `resolveQuietZone(matrixSize)`
- `effectiveQuietZone(matrixSize)`
- directional matrix-aware helpers

Do not reintroduce matrix-less quiet-zone calculations.

## 11. BASIC Geometry

For EF BASIC parity verify:
- finder geometry
- timing/alignment geometry
- data geometry
- scale
- rounded radius
- quiet zone
- SVG coordinate system
- node ordering
- logo ordering

Do not force behavior to match EF unless the EF source demonstrates it.

## 12. Testing Protocol

Run targeted tests first, then:

```powershell
.\gradlew.bat :app:testDebugUnitTest
```

When relevant:

```powershell
.\gradlew.bat :app:lintDebug
```

Then a clean verification:

```powershell
.\gradlew.bat clean :app:testDebugUnitTest
```

Do not report success from stale output.

Report:
- total
- failures
- errors
- skipped
- ignored

If the test count changes, explain why.

## 13. Pre-Commit Gate

Before committing:

```powershell
git status
git diff --check
git diff
```

Review all changed files.

Do not commit unrelated changes.

Re-search the original defect symbols and verify the fix is actually connected to production code.

## 14. Git Rules

Use focused commits, for example:

```text
fix(qr): close EF parity normalization gaps
fix(qr): unify gradient geometry across renderers
test(qr): add BASIC parity regression coverage
```

After pushing:

```powershell
git status
```

The working tree must be clean.

Do not claim remote CI passed unless remote CI actually ran and its result was observed.

A local test result is not a CI result.

## 15. Final Audit Report

Use these sections:
1. FIXES IMPLEMENTED
2. TESTS ADDED
3. TEST RESULTS
4. EF PARITY VERIFIED
5. INTENTIONAL DIVERGENCES
6. REMAINING LIMITATIONS
7. COMMIT / PUSH STATUS

Every parity claim must have an evidence level:
- SOURCE VERIFIED
- CODE VERIFIED
- UNIT TEST VERIFIED
- RASTER VERIFIED
- DECODE VERIFIED
- NOT VERIFIED

Never claim fully closed solely because the existing test suite passes.

## 16. Anti-Patterns

Do not:
- add a helper without wiring it into production code
- add a test that cannot fail when the original bug returns
- claim pixel parity from structural equivalence
- claim EF parity from similar-looking code
- duplicate mode normalization logic
- silently change explicit user customization
- hide rendering failures behind nulls
- invent CI results
- rely on stale test reports
- remove VeilFrame-only features for superficial EF similarity
- mix unrelated refactors into a parity fix

## 17. Definition of Done

A QR task is complete only when:
1. The defect is fixed in production code.
2. The production path actually uses the fix.
3. A regression test directly covers it.
4. Relevant existing tests pass.
5. The full suite passes.
6. No contradictory duplicate implementation remains.
7. EF claims are backed by the appropriate evidence level.
8. Limitations are documented.
9. Git diff is reviewed.
10. The final tree is clean after commit/push.

Optimize for correctness, reproducibility, deterministic behavior, maintainability, and honest verification — not for producing a superficially successful audit report.
