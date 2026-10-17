LiteSpec composable reference archive
====================================

This folder intentionally uses the requested spelling: refrence.

Start with index.html. manifest.json records repository revisions, source paths,
line ranges, hashes, source classification and migration constraints. verification.json
records comparisons against pinned upstream files. licenses/ retains source licenses
and applicable upstream notices.

context/, backbone/, proposal/ and objectives/ contain source excerpts with original
variable names and function bodies. framework/ contains the existing LiteSpec
operations and integration contracts that should be reused.

These excerpts are evidence, not runnable standalone modules. Never import them
from the new model, training, or deployment packages. Missing helpers and imports
remain in the full source paths recorded in manifest.json.

KVShot KV-only, BV and standalone Soft CE have .paper.txt evidence notes rather
than invented author source code. These notes clearly distinguish paper definitions
and proposed adaptations.

New composable implementation conventions:
  naming-conventions.json  - canonical names, shapes and meanings
  naming-map.json          - source-to-integrated variable/parameter mapping
  framework-reuse.json     - reuse decisions and methodology constraints
  reduction-contracts.json- native numerator/denominator and runner behavior
  implementation-plan.json- first complete route and incremental stages
  scaffold-manifest.json  - empty files created in the requested remote checkout

Current scope is reference preparation, docstring-only code organization, and a
refined plan. No model/adapter registration, forward implementation, training or
deployment logic has been integrated. Existing source snapshots remain unchanged.
