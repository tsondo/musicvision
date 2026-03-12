# License Update Instructions

## Files to Add or Replace

| File | Action | Destination |
|------|--------|-------------|
| `THIRD-PARTY-NOTICES.md` | **Add** (new) | Repository root |
| `LICENSE-COMMERCIAL` | **Replace** existing | Repository root |
| `CLA.md` | **Add** (new) | Repository root |
| `CONTRIBUTING.md` | **Replace** existing (or add if missing) | Repository root |
| `vendor-LICENSE.md` | **Add** (new), rename to `LICENSE` | `src/musicvision/video/vendor/LICENSE` |

## pyproject.toml Change

The SPDX identifier is already close but should use the canonical format. In `pyproject.toml`, the license field is currently:

```toml
license = { text = "LicenseRef-PolyForm-Noncommercial-1.0.0" }
```

This is correct — PolyForm Noncommercial does not have an official SPDX short identifier, so `LicenseRef-*` is the right approach. **No change needed here.**

However, add a `license-files` field so tools can find both license files:

```toml
license-files = ["LICENSE", "LICENSE-COMMERCIAL", "CLA.md", "THIRD-PARTY-NOTICES.md"]
```

Add this line directly below the `license = { text = ... }` line.

## Summary of Changes

1. **THIRD-PARTY-NOTICES.md** — Catalogs every upstream model/library with license, commercial-use status, and vendored code boundary statement.

2. **LICENSE-COMMERCIAL** — Three changes:
   - Fixed prices replaced with "contact us" language
   - 30-day refund window added for initial license term
   - Late payment interest reduced from 1.5% to 1.0%
   - §7 now references CLA.md instead of claiming DCO

3. **CLA.md** — Proper Contributor License Agreement modeled on Apache ICLA. Replaces the informal DCO reference that was previously in LICENSE-COMMERCIAL §7.

4. **CONTRIBUTING.md** — Updated to reference CLA.md instead of DCO. Clear statement that contributions may be included in commercial releases.

5. **vendor/LICENSE** — Explicit Apache 2.0 license boundary for vendored code, preventing any ambiguity about PolyForm coverage.
