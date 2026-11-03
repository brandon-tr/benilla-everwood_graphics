# Full Character Names Implementation Plan

> For the implementation worker: follow each self-contained Codex work order, one at a time, through manual copy/paste into DeepSeek Harness Desktop. Do not dispatch agents or automate the handoff. The user's AGENTS.md protocol takes precedence over generic skill execution or commit instructions.

**Goal:** Make a required first and last name the unique, persistent player-facing identity throughout the custom client and realm.

**Architecture:** Keep one canonical `First Last` string in existing name-bearing records and packets, with two inputs on character creation. Character IDs continue owning progress and relationships. Validate each part under its existing naming rules, plus combined restricted-content checks; use exact full-name lookup for players.

**Tech stack:** Rust/Bevy 0.18.1 client; C++14 MaNGOS server; existing SQL schema and migration system. No new dependency is planned.

**Spec:** [Approved design](../specs/2026-10-30-full-character-names-design.md).

## Global constraints

- Empty realm at launch; no legacy-character conversion or destructive database reset.
- Both parts required; initial uppercase letter and remaining lowercase letters in each cased part; existing locale rules apply independently.
- Full name unique per realm, ignoring capitalization; up to 25 characters for the Latin two-part maximum, not 25 UTF-8 bytes.
- Full names displayed everywhere; player lookups never guess from prefixes or first names.
- Name reuse after successful deletion transfers no character-owned state.
- Preserve NPC/pet/guild/channel naming and action permissions, faction gates, visibility, and existing ID-based interactions.
- Custom fork behavior; no copied installation assets, edits to the WoW installation, generic engine refactor, dependency changes, account-wide surname rules, surname-hiding settings, or new rename service.
- No commits, merges, rebases, pushes, database writes, migrations, deployments, or external-service probes without the user's explicit authorization.
- Use one worker per worktree. Do not modify the active TAA worktree. Codex reviews the actual worker worktree, including untracked files, after each returned report.

## Source baseline and workspace boundaries

EXISTING / VERIFIED: client base `c9f15a874f1c5fa38cb96f5796cef1cceb18fbdd`; server base `316e886869ed7fb3a747e2adbcc720e9ada995cf`. Recheck before every work order; later work orders must account for verified uncommitted work from earlier tasks rather than assuming a branch contains it.

Client worker workspace prepared: `C:/Users/dev machine/Desktop/benilla-everwood_graphics-names`, branch `codex/full-character-names`, initially clean. Its tracked AGENTS.md is the earlier manual-handoff version; work orders additionally carry the user's v3 preflight, ownership, escalation, and report requirements. Do not overwrite either instructions file.

Planning workspace: `C:/Users/dev machine/Desktop/benilla-everwood_graphics`, branch `main`. Preserve its preexisting modified AGENTS.md, glossary, design, and plan. These uncommitted documents do not automatically appear in the worker worktree.

Server source workspace: `C:/Users/dev machine/Desktop/core`, branch `development`; preexisting untracked AGENTS.md and two world archive/SQL files are protected. No server worker worktree is assigned yet. Prepare and verify one when its first task is ready.

## Review focus

1. Mixed-case and non-ASCII names: normalization, validation, comparisons, and cache keys must agree; a restart must preserve identity.
2. Concurrent create/delete/rename: never duplicate an active identity, overwrite another character, or release a name after a failed operation.
3. Stock Lua chat parsing: `/w First Last message` must reach the intended recipient rather than treating the last name as message text.
4. Prefix lookup in party/raid passes: an absent or distant player must never be chosen from an abbreviated name; NPC prefix targeting must remain intact.
5. Deleted-name reuse: saved social references and already-addressed mail must remain tied to the original ID, including delayed/offline paths.

Each item has explicit coverage in the owning stage below. These are implementation risks, not verified defects beyond the source facts stated.

## Stage FN-01: client full-name validation foundation

This is the first execution-ready task. It is intentionally not a live cutover and is independently testable without a server or game installation.

**Files owned:** `crates/benilla-ui/src/script/char_name.rs` and `crates/benilla-ui/src/script/mod.rs` only. Keep tests in the existing char_name test module; no new dependency, file, fixture corpus, or struct.

**EXISTING / VERIFIED interface:** `check_char_name(name: Option<&str>, lists: &dyn NameLists, charsets: NameCharsets) -> u8`; `NameLists`; `NameCharsets`; existing `script` re-export. Preserve the existing function's signature, behavior, error precedence, and all 13 existing tests.

**NEW / PROPOSED interface:** `pub fn check_full_char_name(name: Option<&str>, lists: &dyn NameLists, charsets: NameCharsets) -> u8`, publicly re-exported from `benilla_ui::script`. It is a validator, not a case converter or Lua binding.

- [ ] Add synthetic regression tests first. On a nonempty full-name string, split at the first ASCII space; a missing separator is `0x48`. `None` or an empty input is `0x45`. Validate first part then last part with the existing helper; return the first failure unchanged. An empty part therefore gives `0x45`; extra spaces, components, tabs, line breaks, NULs, and punctuation fail the existing character checks.
- [ ] Once both parts pass, call each existing restricted-word-list predicate on the original spaced full name and on the two parts concatenated without the separator. Any hit returns `0x4b`; otherwise return `CHAR_NAME_SUCCESS` (`0x52`). Do not apply single-part length, repetition, or charset selection to the combined string. Do not normalize, trim, truncate NULs, or remove arbitrary whitespace.
- [ ] Implement the smallest helper and export. Add a brief `Deviation:` comment identifying this as custom two-part policy; do not claim it is stock reference behavior. Do not change production callers in this stage.
- [ ] Test `full_name_accepts_two_valid_parts_and_maximum_latin_length`: `Asha Brightvale`, `aSHA brightVALE`, and `Abcdefghijkl Mnopqrstuvwx` succeed; a 13-unit first or last Latin part fails `0x47`.
- [ ] Test `full_name_rejects_missing_parts_and_malformed_separators`: the statuses above, plus a third word, doubled spaces, tab/newline/NBSP separators, and embedded NULs. These must not be silently repaired.
- [ ] Test `full_name_preserves_per_part_errors_and_order`: first-part failures precede last-part failures; invalid characters return `0x48`, too-short `0x46`, triple letters `0x4e`, and restricted parts `0x4b`. Preserve per-part error precedence.
- [ ] Test `full_name_checks_each_list_across_the_boundary`: synthetic NameLists independently trip names_profanity, chat_profanity, and names_reserved for spaced and unspaced subjects. `Bl Izza` is rejected when `blizz` is restricted despite both parts passing alone.
- [ ] Test `full_name_keeps_length_and_repetition_checks_per_part`: maximum parts pass despite combined length, and `Abb Bc` passes with empty word lists even though joining yields a triple at the boundary. Combined checks are restricted-content checks only.
- [ ] Test `full_name_keeps_locale_caps_per_part`: enUS rejects Hangul; koKR accepts `가나다라마바사아 Abcdefgh` and rejects a nine-unit part; zhCN accepts six ideographs per part and rejects seven. Include mixed primary/secondary parts where each independently passes under the same NameCharsets.
- [ ] Run targeted tests, all benilla-ui library tests, formatting, and crate lint with actual exit codes. Report environmental failures; do not repair unrelated tests or widen ownership.

Baseline evidence: `cargo test --offline --locked -p benilla-ui script::char_name::tests -- --nocapture` passed in the planning checkout: 13 passed, 0 failed, exit 0, rustc/cargo 1.98.1. Worker source blobs match that checkout: char_name.rs `4c81e16a0ac4cf9ab573423154f540a7bef071a0`; script/mod.rs `f907f72ac14340b7267823901188e379b773b6f7`. The worker must run its own baseline; the separate worktree build itself has not been tested yet.

Commands for FN-01, from the worker root:

```powershell
cargo test --offline --locked -p benilla-ui script::char_name::tests -- --nocapture
cargo test --offline --locked -p benilla-ui --lib
cargo fmt --all -- --check
cargo clippy --offline --locked -p benilla-ui --all-targets -- -D warnings
git diff --check
git diff --stat
```

Run the targeted suite before editing, demonstrate a meaningful new test fails before implementation, then run the listed after-change checks once. Never treat a zero-test filter as acceptance evidence. Missing offline dependencies are a blocker to that command, not permission for downloads.

## Stage FN-02: establish server validation and persistence contracts

Depends on reviewing FN-01. This stage needs a refreshed, focused work order after the findings below are resolved; it is not permission to invent server helpers now.

**EXISTING / VERIFIED source map:**

- `src/game/ObjectMgr.cpp`: `normalizePlayerName(std::string&, size_t)`, `ObjectMgr::CheckPlayerName(std::string const&, bool)`, `isValidString(...)`, `ObjectMgr::IsReservedName(...)`, and cache load/insert/name-change/delete methods.
- `src/game/ObjectMgr.h`: `MAX_PLAYER_NAME` 12 and `MAX_INTERNAL_PLAYER_NAME` 15; these serve different consumers.
- `src/shared/Util.h`: `wcharToUpper` / `wcharToLower`, including special `ß` / `ẞ` mappings.
- `src/game/Database/DBCStores.cpp`: `ValidateName(std::wstring const&)`, backed by names profanity/reserved regexes. Unlike the client interface, the inspected server path does not check ChatProfanity here or enforce the client's repetition/locale-cap rules.
- `src/game/Chat/Channel.cpp`: `Channel::Channel(...)` calls `normalizePlayerName(m_name, 128)` for custom channels. Do not impose player two-part syntax on this caller.
- `src/game/Handlers/CharacterHandler.cpp`: `HandleCharCreateOpcode`, `HandleCharRenameOpcode`, and `HandleCharDeleteOpcode`; creation and rename opcodes use `PACKET_PROCESS_WORLD` in `src/game/Server/Protocol/Opcodes.cpp`.
- `src/game/Objects/Player.cpp`: `SaveNewPlayer`, `SaveToDB`, load validation, and `DeleteFromDB`. Save paths use `REPLACE INTO`; new-character cache insertion follows transaction submission.
- `sql/characters.sql`: `characters.name`, `characters.deleted_name`, and ticket-creator name columns cap at 12; name index is nonunique; soft deletion clears active `name` to an empty string.
- `sql/migrations/README` and recent migration scripts describe migration conventions; existing build cache uses Visual Studio 18 2026, Release. No project test runner has been verified.

- [ ] Trace creation -> database transaction queue -> completion/error -> active-name cache -> response, including alternate writers (bots, dumps, forced/admin rename, restore). Record thread ownership and failure semantics before selecting uniqueness enforcement.
- [ ] Resolve a server-authoritative locale/charset and restricted-list contract matching client validation, with a shared synthetic fixture matrix. Do not port or copy GPL server implementation into the MIT/Apache client. Do not silently narrow locale support to ASCII.
- [ ] Resolve length-preserving capitalization and case-insensitive equality for accepted non-ASCII inputs. Existing client rank comparison is ASCII-insensitive; server casing can produce characters outside the client's current input set. Specify behavior and tests for these mappings before wiring normalization.
- [ ] Choose the smallest safe name-claim mechanism consistent with actual threading and async persistence. Never add a unique name index alongside `REPLACE INTO`: a conflicting insert could delete another row. Never add a plain unique index on empty soft-deleted names. Concurrent claims, persistence failure, failed rename, failed delete, and restore conflicts need assertions.
- [ ] Inventory all player-name schema columns and byte/character limits, including live schema version only if safely accessible without exposing credentials. Prepare scripts for review; do not apply them. Empty-at-launch does not authorize deleting present local data.
- [ ] Establish an offline server test/build route using installed tools; no network installs, server startup, or migration side effects. Report any missing harness/toolchain explicitly.

These are exact research gates for the next server work order, not open-ended worker architecture discretion.

## Stage FN-03: server full-name lifecycle and routing

Depends on FN-02's reviewed decisions and test route. Use a dedicated server worktree and a fresh symbol-level handoff; split validation/lifecycle and actions into separate review cycles if the diff would otherwise cross too many owners.

### First bounded subtask: FN-03A, normalization and offline tests

The server worker workspace is `C:/Users/dev machine/Desktop/core-names`, branch `codex/full-character-names`, initially clean at server base `316e886869ed7fb3a747e2adbcc720e9ada995cf`. Its instructions also include the original server checkout's `C:/Users/dev machine/Desktop/core/AGENTS.md`; do not copy or overwrite that file.

NEW / PROPOSED: `src/shared/PlayerName.h` and `.cpp`, exposing only `bool normalizeFullPlayerName(std::string& name)`. Keep grammar checks private; there is no need for a separate public splitter or configurable maximum argument. Add the files to the existing `shared_SRCS` list in `src/shared/CMakeLists.txt`.

The helper is a formatting primitive, not admission validation. It requires two nonempty parts with exactly one ASCII space, rejects controls/other whitespace, malformed UTF-8, and embedded NULs, enforces an absolute maximum of 12 UTF-16 units per part, and updates its argument only on success. Per-locale 8/6-unit caps, allowed scripts, minimum length, repetition, word lists, and availability remain the later validator's responsibility.

Reuse inline `wcharToUpper` / `wcharToLower` mappings from `Util.h`; explicitly complete the missing `ÿ` / `Ÿ` pair locally in the new helper. A leading `ß` maps to single-character `ẞ`, not the expanding string `SS`. These are converter outputs, not a decision that the input validator currently admits every canonical counterpart. Client/server canonical-character admission must be resolved before wiring this helper into production. Preserve accents and uncased characters; count supplementary scalars as two UTF-16 units.

Use the already-vendored checked `utf8cpp/utf8.h` iterator/append operations directly, avoiding an unnecessary link of the entire Util.cpp object into the pure test executable. Existing `Utf8toWStr(..., max_len)` truncates when a nonzero cap is supplied; it must not be used to hide excessive input length.

NEW / PROPOSED: standalone C++14 test project `tests/full_player_name/CMakeLists.txt` and `PlayerNameTests.cpp`. Compile the actual production PlayerName.cpp with only standard-library/header dependencies and the existing source include paths; no database libraries, game startup, test-framework installation, or root CMake option changes. Build output belongs under ignored `build/full-player-name`. Register one CTest test named `full_player_name`; its executable must perform explicit checks and return nonzero on failure even under Release/NDEBUG.

Test shape, unchanged input after every failure, case conversion of both parts, Latin-1 and canonical counterpart pairs, uncased Hangul/CJK, UTF-16 boundary counts including supplementary scalars, invalid UTF-8/NUL/control/whitespace, idempotence, and independently allocated calls. Retain the old normalizer unchanged. No runtime callers, schema changes, or database work are part of FN-03A.

### Next bounded subtask: FN-03B, client canonical admission

The user chose to reject names whose required capitalization leaves the existing character set. Keep FN-03A's formatter behavior: conversion is not admission. Do not add `ẞ`, `Þ`, or `Ÿ` to the client alphabet or change stock single-name checks.

NEW / PROPOSED: in `crates/benilla-ui/src/script/char_name.rs`, add `pub fn prepare_full_char_name(name: Option<&str>, lists: &dyn NameLists, charsets: NameCharsets) -> Result<String, u8>`, re-exported through `script/mod.rs`. First call the existing `check_full_char_name` on the raw form and forward its error unchanged. Capitalize each independently valid part, joining with exactly one ASCII space. Use single-character `ß` -> `ẞ`, not Unicode's expanding `SS`; other admitted Latin-1 casing follows the normal one-character uppercase/lowercase mappings, with Hangul/CJK unchanged. Then call the existing full-name validator on the canonical result and return its error or the validated canonical string.

Both raw and canonical validation include per-part and combined word-list checks. This also catches repetition and restricted-content failures introduced by capitalization. For example, raw alternating accented letters can become three identical lowercase letters within a part. Never retry with raw text or another spelling if canonical admission fails.

Preserve `check_char_name`, `check_full_char_name`, and all existing tests and signatures. The new preparation function is the future creation/rename admission entry point; the existing check functions validate the supplied representation. Do not wire any runtime callers in this bounded task. Own only the existing client `char_name.rs` and `script/mod.rs`; acknowledge and preserve FN-01's verified uncommitted changes in that worktree's freshness gate.

Test canonical ASCII/accented outputs, rejection of each exceptional initial in either part, allowed internal lowercase forms, raw-error forwarding, canonical-only repetition/list failures, locale boundaries, and maximum lengths. Re-run the scoped client checks. Later server admission must match these outcomes before either side is enabled.

### Next bounded subtask: FN-03C, server admission parity

Extend the existing server PlayerName module and its offline harness; no production callers or database changes. NEW / PROPOSED interfaces in PlayerName.h:

- `struct PlayerNameCharsets { std::uint8_t primary; std::uint8_t secondary; static PlayerNameCharsets ForClient(std::int32_t region, std::size_t locale); };`
- Abstract `class PlayerNameLists`, with const boolean `NamesProfanity`, `ChatProfanity`, and `NamesReserved` methods accepting `std::string const&`, plus a virtual default destructor. There is no production no-lists default.
- `std::uint8_t checkFullPlayerName(std::string const&, PlayerNameLists const&, PlayerNameCharsets);`
- `std::uint8_t prepareFullPlayerName(std::string&, PlayerNameLists const&, PlayerNameCharsets);`

The charset factory exactly matches the client's region/locale tables. The checker matches raw client full-name semantics, including UTF-16 unit handling, per-part error precedence, ASCII-only triple folding for charset 1, locale-specific caps, all three per-part/spaced/concatenated restricted-content checks, and existing status bytes. Reject malformed UTF-8 with INVALID_CHARACTER rather than throwing; valid UTF-8 whose part needs 1024 or more UTF-16 units returns FAILURE before the naming walk. Keep decoding storage bounded to the 1023 usable units while validating the complete reached part's encoding.

The preparation function checks raw input, formats a temporary copy with normalizeFullPlayerName, checks the canonical spelling, and swaps into the argument only on success. Preserve original bytes on every failure, including canonical rejection. No alphabet expansion, speculative persistence API, configurable policy, root CMake changes, or DBC loader changes.

For Hangul, generate a private 175-word bitmap in `src/shared/PlayerNameHangul.h` from the already-installed Python standard library EUC-KR codec: lead bytes B0..C8 inclusive and trail bytes A1..FE inclusive. These yield the required 2,350 KS X 1001 syllables. Codex independently verified all 175 generated words equal the inspected client bitmap; require the worker to repeat this comparison against the pinned read-only client source. No install assets, external download, runtime Python dependency, or committed generator script is needed. Test all Hangul-block candidates, endpoints/unlisted syllables, region overrides, Latin-1 edges, koKR ASCII cap 8, zhCN Latin cap 12, and raw/canonical parity cases from FN-01/FN-03B.

Future runtime work must supply an explicit validated profile and a real three-list DBC adapter. This pure task does not resolve the transport/configuration choice for the profile or load ChatProfanity.dbc. Test callbacks exercise the complete contract without a running server.

### Storage prerequisite FN-04A: capacity scripts only

Issue this work only after FN-03C's acceptance evidence is complete, including independent canonical-only coverage of each of the three word-list predicates. A passing aggregate test count does not replace that fixture-level review.

Own only `sql/characters.sql` and new `sql/migrations/20261030222040_characters.sql` in the server names worktree. Widen exactly `characters.name`, `characters.deleted_name`, and `gm_tickets.name` from varchar(12) to varchar(25), preserving their existing nullability, defaults, comments, table engine/charset/collation declarations, and indexes. No unique index, nullable active-name change, data update, name-claim table, or runtime code belongs in this prerequisite.

Use the repository's guarded add_migration procedure form. Deliberately record migration ID `20261030222040` only after both widening ALTER TABLE statements, so a DDL failure cannot falsely mark the migration applied. This is a narrow exception to the old README/template's insert-first order: the inspected CMake migration catalog reads filenames, and Database::CheckRequiredMigrations checks registered IDs; neither parses the first SQL line. Widening the same three definitions again is safe if an earlier attempt completed only part of the work; no rollback or multi-statement atomicity is claimed.

This prepares source artifacts only. Do not invoke a SQL client, connect to any database, import a schema, apply the migration, regenerate migration headers, configure the root server project, start the server, or install anything. Review the source diff and static assertions for the three definitions, preserved attributes, guarded execution, matching ID, and record-last ordering. Runtime DDL, repeat-run behavior, SQL modes, and compatibility with a deployed customized schema remain NOT RUN/UNVERIFIED until a separately authorized disposable database check and deployment preflight.

The source file predates some later migrations; do not repair its unrelated columns. The script targets the three inspected definitions and requires live-schema review before application. Adding uniqueness is explicitly deferred until REPLACE INTO and persistence/cache-order hazards are addressed in the same reviewed change set.

### Runtime-input gate FN-04B: real lists and profile contract

Before implementing the production adapter, prove the restricted-pattern matcher and settle the naming-profile inputs from source. Actual read-only tables are available under `C:/Users/dev machine/Desktop/core/bin/Release/5875/dbc`: NamesProfanity has 1,535 nonempty rows, NamesReserved 2,288, and ChatProfanity 2,289. All are two-field, eight-byte-record WDBC tables. Do not copy them into the repo or change the installation.

The client `text_filter.rs` defines ASCII-only case folding and a custom word class: ASCII letters/digits/underscore plus its Latin-1 letter set, with code points above FF non-word. Both `\\<` and `\\>` mean a nondirectional word boundary. The existing server maps them to start/end anchors and does not load ChatProfanity, so its current matcher cannot be assumed equivalent. A small offline probe may investigate installed std::basic_regex with explicit traits for ASCII-only folding and word classification; it must compile all 6,112 actual patterns and prove boundary/case behavior rather than silently dropping unsupported restrictions. No new regex engine/dependency or production matcher is authorized by this discovery task.

Trace the client's Wow.ini Region/locale -> VM NameCharsets and locale-bearing logon request, then server account locale -> WorldSocket::_HandleAuthSession -> WorldSession constructor/fallback fields. Session DBC availability fallback must not silently change naming policy. Resolve how the effective full-name profile will agree between client and server without guessing enum-index equivalence or introducing an unreviewed packet change. Existing identities and lookups must remain usable across client locale changes; do not apply the actor's creation profile to another player's existing name.

FN-04B is targeted discovery with optional throwaway standard-library probes under ignored build/naming-probes only. It must leave all tracked and existing untracked source/SQL/test artifacts unchanged, make no database/network/service calls, and return concrete matcher/profile interfaces and acceptance evidence for the next runtime implementation handoff.

### Corrections and limits from reviewing FN-02

- Two ordinary create handlers are world-thread serialized and each inserts its cache entry before returning. SQL queuing alone does not let a second ordinary create miss that in-memory claim. Ghost claims after failed SQL and races involving other writers remain real concerns.
- zhCN/zhTW use the Latin secondary charset with a 12-unit cap; the ASCII secondary charset capped at 8 is koKR's. Do not copy the discovery report's mistaken zhCN ASCII-cap statement into implementation.
- A checked single-row name claim/release does not by itself make multi-table creation or deletion safe on MyISAM. The report's nullable-name/unique-index proposal is not a complete failure-safe lifecycle design yet.
- Before accepting an ordinary-save SQL rewrite, prevent stale queued saves from undoing explicit rename or deletion; a direct rename alone does not establish ordering against already-enqueued writes. Do not introduce a general database-threading redesign without a separate reviewed decision.
- The independent baseline `game` build passed and `ctest -N` found zero tests in the original server checkout. No live database engine, collation, migration state, or configured runtime values were verified.

### Subsequent lifecycle work

- [ ] Implement the agreed per-part validation, capitalization, restricted-content checks, canonical lookup keys, and storage capacity. Preserve separate non-player normalization for channels and unrelated names.
- [ ] Update creation, forced rename, login validation, startup cache load, and retained admin/bot/import/restore paths to the same contract. No first-name-only entry point may create an active character or bypass full-name lookup.
- [ ] Implement and test safe availability/claim/release behavior based on the reviewed persistence design. Preserve character ID allocation and all ownership/security checks.
- [ ] Update actual player-name consumers in `Handlers/ChatHandler.cpp`, `GroupHandler.cpp`, `GuildHandler.cpp`, `MailHandler.cpp`, `ChannelHandler.cpp`, `ObjectAccessor.cpp`, and `Chat/Chat.cpp`. `ChatHandler::ExtractPlayerNameFromLink(char**)` and `ExtractPlayerTarget(...)` need two-part plain-name grammar without breaking hyperlink or selected-ID forms.
- [ ] Audit `HandleTeleportToUnitOpcode(...)` in `Handlers/MiscHandler.cpp`, whose current byte-length check and generated `.goname` command assume shorter names. Preserve permission checks and prevent command-argument confusion.
- [ ] Prove routing with repeated first names, offline mail, mixed-case queries, all active word lists, maximum UTF-8 names, unauthorized operations, and no prefix fallback. Prove two concurrent claims yield at most one character and a failed save cannot replace another character.
- [ ] Prove both deletion modes release names safely and restore cannot take an already-reused name. Verify old social/mail references remain ID-based. Preserve GUID generation; do not reuse IDs as part of name recycling.

## Stage FN-04: client creation, rename, and packet round trips

Depends on FN-01 and the reviewed server contract. Exact UI adapter edits need a refreshed work order after reading the player's stock GlueXML through the existing read-only loader.

**EXISTING / VERIFIED anchors:** `script/glue_create.rs` registers `CreateCharacter` and `GetRandomName`; `script/glue_select.rs` registers `RenameCharacter`; `GlueCreateRequest.name` and protocol `CharCreateReq.name` are strings. `benilla-app/src/ui_script/glue_mgr.rs` contains `create_character`, `rename_character`, and `run_step`. `ui_script/glue.rs::load_glue` / `load_glue_layer` currently load only `AccountLoginRealmlist.xml` after stock GlueXML. `manifest.rs` and layer tests pin shipped UI structure.

- [ ] Use the fork's existing UI loading mechanism to provide required first/last fields, labels, Tab/Shift-Tab focus order, Enter-to-submit behavior, and field-specific correction. Preserve appearance customization, selection, delete confirmation, and connection-state handling.
- [ ] Keep the existing one-name-string request and wire structure: concatenate validated parts, send one NUL-terminated full name, and preserve it through enumeration, rename, login, and player-name-query responses. Add maximum UTF-8 round trips and malformed-input tests.
- [ ] Use the same authoritative capitalization contract as the server. Never use Unicode uppercase expansion accidentally or validate only raw input then persist an invalid normalized name.
- [ ] Update retained forced rename UI, random-name button if present, dev/probe creation helpers, and directly necessary fixtures to supply two valid parts. Do not mutate WoW installation files or copy its FrameXML/GlueXML into the repo.
- [ ] Prefer adapting already-shipped original UI assets and existing loaders. If an additional shipped adapter is essential, name it in the refreshed handoff and explicitly approve the fork deviation and affected manifest tests; do not grow the layer silently.
- [ ] Test successful creation, rejection retaining both inputs, reconnect/reload, random-name output if retained, and long-name layout. Record install/UI checks as NOT RUN when unavailable.

## Stage FN-05: exact client player actions and chat grammar

Depends on FN-03/FN-04 contracts. This stage must exercise stock Lua entry points, not just dev/probe parsing.

**EXISTING / VERIFIED anchors:** `ui_chat/input/parse.rs::parse_line` / `slash_target_name` are primarily dev/probe paths; their header explicitly says stock `ChatEdit_ParseText` owns built-in typed commands. `script/chat_send.rs` registers `SendChatMessage`, whose target is a separate string argument. `script/unit/by_name.rs::resolve` handles party/raid passes and held candidates; `Held.is_player` distinguishes players. `benilla-app/src/target/by_name.rs::NameSearch` feeds that resolver. `NameCache::resolve_player` and `resolve_unit` resolve by GUID.

- [ ] Adapt the stock chat entry flow so `/w First Last message` consumes two recipient words before message text. Preserve `/r`, click-to-whisper, unit menus, and hyperlink targets. Missing recipient components or message sends nothing. Do not rely on changing dev `parse_line` alone.
- [ ] Require exact full-name matching for player candidates in party, raid, and held-object passes, using the reviewed case policy. Keep NPC/pet prefix behavior and existing permission/selection filters. Test a near prefix candidate against a distant exact player and an unheld party member.
- [ ] Trace all player-name actions, including emotes, duel, target, assist, follow, social, guild, channel moderation, and mail; preserve selected-ID actions and require complete names for string inputs. Never relax checks for addons.
- [ ] Test typed Lua UI execution and direct API calls, duplicated first names, mixed case, no-result cases, replies after reload/reconnect, and offline recipients. Add negative assertions that no packet/action reaches a player from a partial name.

## Stage FN-06: full-name display and compatibility sweep

Depends on FN-04/FN-05. Most displays should consume the existing full-name string rather than new per-widget surname fields.

- [ ] Audit actual callers of NameCache and player-name responses. Check selection, overhead labels, nameplates, chat, party/raid frames, tooltips, friends/ignore/guild lists, mail, auctions, combat output, item attribution, and system messages.
- [ ] Fix only observed name truncation, input caps, hyperlink parsing, and layout assumptions. Preserve full identity even when a layout needs wrapping or more width. No generic widget redesign or surname-hiding setting.
- [ ] Use two players sharing a first name and maximum-length Latin/non-ASCII names in UI tests. Verify labels, click targets, sorting, and links. Inspect already-loaded third-party addon behavior, documenting incompatible addons rather than promising arbitrary addon compatibility.
- [ ] Audit custom-client admission as an existing deployment constraint. A larger accepted name is not proof that stock clients are excluded; changes to authentication/security policy require a separate explicit decision.

## Stage FN-07: combined verification and integration review

Depends on all implementation stages being independently reviewed.

- [ ] Run scoped client `scripts/check.sh` and the applicable `scripts/gates.sh` checks on the actual proposed landing tree, using the installed supported shell/toolchain and respecting no-install skip rules. Run server build and tests via the route established in FN-02. Record exact exit codes and platform coverage.
- [ ] After explicit authorization for a disposable test database/server and schema application, create shared-first/shared-last characters, test duplicates and concurrent claims, log in, exercise chat/social/mail/group/guild actions, log out, restart, and repeat lookups.
- [ ] Verify deletion and name reuse with fresh IDs, delayed mail, offline social references, both deletion modes, failed/unauthorized deletion, and restore conflict behavior. Never run against production or delete user characters to manufacture an empty baseline.
- [ ] Follow `.probe-identity` rules with distinct test accounts for concurrent clients, `WOW_UNATTENDED=1`, `WOW_NOSOUND=1`, and relevant GM settings. No unattended combat probes or account kicking.
- [ ] Review the final combined tree with TAA changes only after explicit integration authorization. Recheck shared glue/app/config files; no source or resource overlap is assumed merely from using different branches.
- [ ] Report implemented, independently verified, NOT RUN checks, remaining risks, and committed/merged/pushed status separately. No completion claim until the full approved acceptance matrix has evidence.

## Delivery rule

Issue only FN-01 now. The remaining stages are a dependency roadmap, not execution-ready worker prompts. Each next handoff is written from the inspected results and a fresh baseline, with exact owned files, signatures, tests, and any necessary decisions resolved before editing. After printing a handoff, stop and await the user's returned report.
