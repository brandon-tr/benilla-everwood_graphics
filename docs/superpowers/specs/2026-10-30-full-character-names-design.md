# First and last character names

The user approved proceeding from this design on 2026-10-30. Implementation remains a manual DeepSeek Harness Desktop handoff, followed by independent Codex review. This does not authorize commits, integration, deployment, or executing schema changes.

## Intent and agreed behavior

This custom client and realm use first and last names as the actual player-facing character identity. Players can share either part without ambiguity because every player-name action uses the complete combination.

- The realm starts empty. There is no existing-character migration requirement.
- Only the custom client is supported. Stock 1.12.1 client compatibility is outside this feature.
- Character creation requires separate first-name and last-name fields.
- Each part follows the existing character-name validation rules, including locale-dependent character sets, length limits, repetition restrictions, and restricted-word lists.
- Check restricted words across both parts too; separating a forbidden word between fields must not bypass validation.
- Capitalize the first letter of each part and lowercase the rest. Input capitalization does not create a distinct name.
- Both the entered form and its capitalized form must satisfy the existing allowed character sets and naming rules. Do not admit additional uppercase counterparts: a part starting with `ß`, `þ`, or `ÿ` is rejected when capitalization produces the currently disallowed `ẞ`, `Þ`, or `Ÿ`. Existing lowercase occurrences inside a part remain governed by the existing rules. Keep the stock single-name validator unchanged.
- The canonical full name is `First Last`, with exactly one ASCII space between its two parts. Neither part contains spaces or punctuation under the agreed character-name rules.
- The full combination is unique per realm, ignoring capitalization. First names and last names are individually reusable.
- Display the full name wherever a player name appears.
- Require the complete name for typed player actions: whispers, invitations, mail, friends, ignores, guild actions, targeting, assisting, following, and other player lookups. Never select a player using a first name or a prefix, even when there is only one candidate.
- Clicks, contextual actions, replies, and already-selected units continue to identify the intended character without retyping its name.
- Deletion immediately releases the full name. Reuse must not transfer progress, items, mail, friends, ignores, guild membership, or other relationships from the old character.

## Chosen approach and alternatives

Chosen by the user: keep one canonical full-name string in existing name-bearing storage and packets. Split it only where the creation or existing rename interface needs the two inputs. Internal character IDs remain authoritative for relationships and actions already addressed by ID.

Alternative considered: add independent first-name and last-name fields throughout database records, packets, and callers. This makes the parts independently available but widens changes without serving an agreed requirement. It was not selected.

A cosmetic surname would not meet the agreed identity or addressing requirements and is excluded.

For Latin names the full-name capacity must accommodate 12 + 1 + 12 = 25 characters. Existing locale limits apply independently to each part; 25 bytes is not a safe UTF-8 capacity limit.

## Inspected source facts

These observations describe the inspected source, not completed implementation.

- EXISTING / VERIFIED: client repository `C:/Users/dev machine/Desktop/benilla-everwood_graphics`, branch `main`, HEAD `c9f15a874f1c5fa38cb96f5796cef1cceb18fbdd`. `AGENTS.md` was already modified before this design work.
- EXISTING / VERIFIED: server repository `C:/Users/dev machine/Desktop/core`, branch `development`, HEAD `316e886869ed7fb3a747e2adbcc720e9ada995cf`. Preexisting untracked files include `AGENTS.md`, `world_full_14_june_2021.7z`, and `world_full_14_june_2021.sql`; they were not modified.
- EXISTING / VERIFIED: `crates/benilla-ui/src/script/char_name.rs` defines `check_char_name(name: Option<&str>, lists: &dyn NameLists, charsets: NameCharsets) -> u8`. Its character wrapper accepts no extra spaces or apostrophes, checks word lists, and applies charset-specific UTF-16-unit limits.
- EXISTING / VERIFIED: `crates/benilla-ui/src/script/glue_create.rs` defines `GlueCreateRequest` with one `name: String` and registers `CreateCharacter`.
- EXISTING / VERIFIED: `crates/benilla-protocol/src/messages/roster.rs` defines `CharCreateReq` with one `name: String`.
- EXISTING / VERIFIED: `crates/benilla-protocol/src/messages/client.rs` defines `char_create(req: &CharCreateReq) -> Vec<u8>` and `messagechat_whisper(language: u32, target: &str, message: &str) -> Vec<u8>`. Both serialize the relevant name as a NUL-terminated string, rather than separating at spaces.
- EXISTING / VERIFIED: `crates/benilla-ui/src/script/unit/by_name.rs` defines `rank(query: &str, name: &str, dist2: f32, mode: Match) -> Option<Rank>` and `resolve(...)`. `Match::PrefixOk` permits prefix matches. `Held` identifies player candidates with `is_player`; party and raid candidates take their own resolver pass.
- EXISTING / VERIFIED: `crates/benilla-app/src/target/by_name.rs` defines `NameSearch` and feeds the shared resolver for targeting, assisting, and following.
- EXISTING / VERIFIED: server `src/game/Handlers/CharacterHandler.cpp`, `WorldSession::HandleCharCreateOpcode(...)`, normalizes, validates, checks reserved names, and checks name availability.
- EXISTING / VERIFIED: server `src/game/ObjectMgr.cpp` defines `normalizePlayerName(std::string& name, size_t max_len)` and `ObjectMgr::CheckPlayerName(std::string const& name, bool create)`. Normalization currently uppercases the first character and lowercases all remaining characters.
- EXISTING / VERIFIED: server `src/game/ObjectMgr.h` defines `MAX_PLAYER_NAME` as 12 and `MAX_INTERNAL_PLAYER_NAME` as 15.
- EXISTING / VERIFIED: server `src/game/Server/Packets/Character.cpp`, `WorldPackets::Character::CharCreate::ReadFromWorldPacket(...)`, reads one name string. `src/game/Server/Packets/Chat.cpp`, `WorldPackets::Chat::ChatMessage::ReadFromWorldPacket(...)`, reads the whisper target separately from its message.
- EXISTING / VERIFIED: server `sql/characters.sql` declares `characters.name` and `characters.deleted_name` as `varchar(12)`. The shown `idx_name` index is not unique.
- EXISTING / VERIFIED: server `src/game/Objects/Player.cpp`, `Player::DeleteFromDB(...)`, has a soft-delete path that copies `name` to `deleted_name` and clears `name`.
- UNVERIFIED: deployed database version and schema, complete name-column and buffer inventory, all slash-command and Lua parsers, server build/test availability, creation concurrency behavior, deletion/cache ordering, and compatibility with third-party addons.
- UNVERIFIED: how the server currently enforces a custom-client-only policy. Supporting that policy is a constraint, not evidence that enforcement already exists.

## Proposed implementation contracts

The following details are proposed for design review, rather than additional user decisions already made.

### Creation, validation, and persistence

Validate both input parts, normalize each separately, check restricted content across their joined and directly concatenated forms, then construct the canonical full name. Cross-part content checks must not apply a single-part length cap to the entire full name. Client errors should identify the offending field where possible; a server rejection must preserve both inputs for correction.

The server independently enforces the same accepted-name contract. UI validation alone is insufficient. Reuse inspected validators and word-list mechanisms where compatible; audit differences between client locale rules and server configuration before specifying exact helpers. A formatter may produce a disallowed canonical character; only the admission pipeline may accept a name, after validating both the original and the canonical forms. Do not transliterate, expand `ß` into `SS`, preserve an initial lowercase letter as a workaround, or expand the alphabet.

Capacity changes must cover all player-name storage, cache, packet, import/export, administrative, and length-check surfaces, including deleted-name storage. Do not mechanically raise a single-part cap and thereby allow a 25-character first name.

Two simultaneous attempts at the same canonical name must produce at most one character. Determine the smallest safe enforcement mechanism from the actual creation and persistence flow. Do not add a naive unique index on `name`: soft-deleted rows can share an empty name.

Creation and login must work after a server restart. Existing forced-rename or administrative creation paths must obey the same two-part contract if retained; no new player rename service is requested.

### Addressing and display

For `/w Asha Brightvale Hello there`, treat the first two name words as the full recipient and the remaining text as the message. Never retry a failed lookup using a first name. Missing parts or missing message text must fail without sending anything. Replies and clickable names preserve the actual sender identity.

Commands, Lua functions, contextual actions, and server handlers must agree on complete names. Existing APIs already receiving a name string should receive `First Last`; avoid adding a second surname argument everywhere.

Player candidates require exact full-name matching in held objects and party/raid passes. Preserve NPC and creature prefix targeting. Preserve existing permissions, visibility, faction, distance, and action-specific restrictions.

Full names must survive UI input limits, hyperlink encoding, chat parsing, list sorting, and rendering without becoming a different identity. Audit character selection, overhead labels, nameplates, chat, tooltips, party/raid frames, guild and social lists, mail, auctions, combat text/logs, item attribution, and system messages. Do not implement surname-hiding settings.

### Deletion and reuse

After successful deletion, invalidate the old character's active name lookup and make the combination available. Failed or unauthorized deletion must not release a live character's name.

Stored relationships continue to refer to IDs. New typed mail addressed to a reused name resolves to its current owner, while previously delivered or queued mail remains bound to its original recipient ID. Character restoration, if retained, must not steal a name from a newer character.

## Acceptance evidence required

1. Create `Asha Brightvale`, `Asha Moonbrook`, and `Mira Brightvale`; all succeed. A second `asha BRIGHTVALE` fails, including concurrent creation attempts.
2. Enter mixed capitalization; creation, selection, login, queries, and displays consistently return `Asha Brightvale`.
3. Independently reject missing, too-short, too-long, invalid-character, forbidden-repeat, and restricted names in either field. Include locale-specific boundaries and direct server requests bypassing UI validation.
4. Accept valid maximum-length parts and round-trip the full name without truncation. Reject malformed UTF-8, embedded NULs, extra components, and invalid separators at the relevant trust boundary.
5. Reject a restricted word split across first and last names. Preserve legitimate names and independently valid maximum-length parts when running combined checks.
6. Route whisper, invite, mail, social, and guild actions to the exact full-name owner. First-name-only and prefix inputs never act on a player. Test offline mail and persisted social references too.
7. Target, assist, and follow exact player names in nearby and party/raid cases. A matching player prefix is never selected; existing NPC prefix targeting still works.
8. Inspect the named UI surfaces with two characters sharing a first name and with maximum-length names. Verify clicks, replies, and hyperlinks address the correct ID.
9. Delete one character and recreate its full name under a new ID. The new character receives none of the old character's relationships, possessions, or previously addressed mail. Failed deletion does not free the name.
10. Restart the server and log in again; names, uniqueness, and correct lookups persist.
11. Run applicable client and server checks independently, then run combined integration checks. Record unavailable checks as NOT RUN; no result has been established yet.

## Boundaries and delivery

This is custom fork behavior, not a claim about stock 1.12.1. The upstream fidelity rules remain applicable to unrelated work. No realmless identity system, account-wide surname ownership, name reservation service, surname-hiding setting, new rename service, or stock-client support is requested.

The implementation remains a manual DeepSeek Desktop handoff. No implementation worker has run. Before a work order, complete targeted symbol/caller discovery, confirm the client and server worker workspaces and their fresh baselines, and establish file ownership separate from the active TAA work.

Schema changes are necessary, but no database writes, migrations, dependency installation, commits, merges, pushes, or deployment have been authorized or performed here. Schema scripts can be prepared for review; executing them requires explicit user approval.

## External reference

Blizzard's [Create a Name of Your Own in WoW: Forever](https://news.blizzard.com/en-us/article/24304161/create-a-name-of-your-own-in-wow-forever) describes reusable name parts, full-combination uniqueness, and combined naming-policy checks. This feature deliberately uses realm-level uniqueness and always-visible full names as agreed in this design, rather than copying every Forever behavior.
