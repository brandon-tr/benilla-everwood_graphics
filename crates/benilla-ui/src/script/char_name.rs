//! The client's name validator `0x6c96c0` and its two wrappers: the character name check
//! `0x5abd50` (`0x6c9ae0` read as a `CHAR_NAME_*` status, `+0x45`), which the rename's send
//! (`0x46b36f`) and the create screen (`0x4711e2`, `0x4719ab`) call, and the guild name check
//! `0x6c9b70`, which the two charter verbs call through `0x4f5160`.
//!
//! The validator takes the client's locale, a list of extra characters (`edx`) and a flag per word
//! list; it answers a code with the name's unit count and the charset it chose, which each wrapper
//! caps. `0x6c9ae0` passes no extra characters (`0x6c9af5 xor edx,edx`) and all three lists
//! (`0x6c9ae6`-`0x6c9aea`); `0x6c9b70` passes a space alone (`0x6c9b88`) and the two profanity
//! lists (`0x6c9b76`-`0x6c9b7a`). The validator takes two charsets from `0x6c9a70`
//! ([`NameCharsets`]), and the first unit in either decides which one the whole name is held to.
//!
//! The checks, in the order the bytes make them, which decides the code of a name failing two:
//!
//! 1. A null name is `NO_NAME` (`0x6c96d6`).
//! 2. The UTF-8 is converted to UTF-16 into 0x400 units, the NUL included; a name that does not fit
//!    is `FAILURE` (`0x65b350` → `0x6c96fe` → `0x6c9a5f`).
//! 3. An empty name is `NO_NAME` (`0x6c9750` → `0x6c98a4`, the count still 0).
//! 4. Then one walk over the units, left to right (`0x6c9756`-`0x6c9897`), so the first offending
//!    unit decides. Until a unit has picked the name's charset, each is tried against the primary,
//!    then the secondary (`0x6c9764`-`0x6c97e3`), and the one that takes it is the name's
//!    (`0x6c9778`, `0x6c97eb`). A unit in the name's charset, from the third unit on, ends
//!    `THREE_CONSECUTIVE` when the last three are equal (`0x6c977a`-`0x6c97d2` → `0x6c98d4`):
//!    under the C library's `tolower` (`0x40f92b`, ASCII only) for charset 1 (`0x6c9788`), exactly
//!    for any other (`0x6c9874`-`0x6c9886`). Any other unit must be an extra character, else it is
//!    `INVALID_CHARACTER` (`0x6c9803` → `0x6c9827`). An apostrophe first is `INVALID_APOSTROPHE`
//!    and a second one `MULTIPLE_APOSTROPHES` (`0x6c9833`-`0x6c984f`); a space first is
//!    `INVALID_SPACE` and one right after a space `CONSECUTIVE_SPACES` (`0x6c9851`-`0x6c9872`).
//! 5. One unit is `TOO_SHORT` (`0x6c989f` → `0x6c98ac`); from two on, an apostrophe last is
//!    `INVALID_APOSTROPHE` and a space last `INVALID_SPACE` (`0x6c98e0`-`0x6c98f9`).
//! 6. The lists the flags turn on, NamesProfanity, ChatProfanity, NamesReserved (`0x6c9909`,
//!    `0x6c9977`, `0x6c99e1`): any match is `RESERVED` (`0x6c9a51`). `MIXED_LANGUAGES` and
//!    `PROFANE` are never produced.
//! 7. The wrapper's cap, checked last: `0x6c9ae0` refuses over the cap of the name's charset as
//!    `TOO_LONG` (`0x6c9b04`-`0x6c9b27`), 12 units for charset 1, else 8, and charset 3 at most 6;
//!    `0x6c9b70` over 24 units whatever the charset (`0x6c9b9e`).

mod hangul;

/// The three lists the validator can search, each row a PCRE compiled caseless (the shipped
/// `NamesProfanity.dbc`, `ChatProfanity.dbc` and `NamesReserved.dbc`), matched unanchored against
/// the name as typed, in UTF-8 (`0x71fc70` with `edx` the name, `0x6c9925`). The validator takes a
/// flag per list (`0x6c9909`, `0x6c9977`, `0x6c99e1`), so each is asked on its own.
pub trait NameLists {
    /// Whether a row of `NamesProfanity` (the list at `0xce8948`) matches `name`.
    fn names_profanity(&self, name: &str) -> bool;
    /// Whether a row of `ChatProfanity` (`0xce893c`) matches `name`.
    fn chat_profanity(&self, name: &str) -> bool;
    /// Whether a row of `NamesReserved` (`0xce8930`) matches `name`.
    fn names_reserved(&self, name: &str) -> bool;
}

/// No lists, for a VM whose host seated none: nothing is reserved.
pub struct NoNameLists;

impl NameLists for NoNameLists {
    fn names_profanity(&self, _: &str) -> bool {
        false
    }

    fn chat_profanity(&self, _: &str) -> bool {
        false
    }

    fn names_reserved(&self, _: &str) -> bool {
        false
    }
}

/// `CHAR_NAME_SUCCESS`, `0x45 + 0xd`, the one status the callers act on (`0x46b374`).
pub const CHAR_NAME_SUCCESS: u8 = 0x52;

/// The validator's codes (`0x6c96c0`'s returns, `ret 0x18`), before the `+0x45`.
pub(crate) const NO_NAME: u8 = 0;
pub(crate) const TOO_SHORT: u8 = 1;
const TOO_LONG: u8 = 2;
const INVALID_CHARACTER: u8 = 3;
/// `MIXED_LANGUAGES` (4) and `PROFANE` (5) are codes the callers map, which the validator never
/// returns.
pub(crate) const MIXED_LANGUAGES: u8 = 4;
pub(crate) const PROFANE: u8 = 5;
pub(crate) const RESERVED: u8 = 6;
const INVALID_APOSTROPHE: u8 = 7;
const MULTIPLE_APOSTROPHES: u8 = 8;
const THREE_CONSECUTIVE: u8 = 9;
pub(crate) const INVALID_SPACE: u8 = 0xa;
pub(crate) const CONSECUTIVE_SPACES: u8 = 0xb;
const FAILURE: u8 = 0xc;
pub(crate) const SUCCESS: u8 = 0xd;

/// The conversion buffer, in UTF-16 units with the NUL (`0x6c96eb push 0x400`).
const UNITS: usize = 0x400;

/// Charset 1's longest name (`0x6c9b07 mov ecx,0xc`); the others' is [`OTHER_MAX_UNITS`].
const LATIN_MAX_UNITS: usize = 12;
/// Any other charset's longest name (`0x6c9b0e mov ecx,0x8`).
const OTHER_MAX_UNITS: usize = 8;
/// Charset 3's, CJK ideographs, is cut to this (`0x6c9b13`-`0x6c9b1d`).
const CJK_MAX_UNITS: usize = 6;
/// A guild name's longest, whatever its charset (`0x6c9b9e cmp [ebp-0x8],0x18`).
const GUILD_MAX_UNITS: usize = 24;

/// The apostrophe and the space, the two extra characters the validator rules on.
const APOSTROPHE: u16 = 0x27;
const SPACE: u16 = 0x20;

/// Which lists a wrapper turns on, the validator's last three arguments in its order.
#[derive(Clone, Copy)]
struct Searched {
    names_profanity: bool,
    chat_profanity: bool,
    names_reserved: bool,
}

/// What `0x6c96c0` answers: its code, and through its two pointers the units it walked and the
/// charset it chose, which the wrappers' caps read (`[ebp-0x8]` and `[ebp-0x4]` of `0x6c9ae0`).
struct Verdict {
    code: u8,
    units: usize,
    charset: u8,
}

/// The charsets a name is checked against: the ids the validator's dispatch (`0x6c9dd0`) takes.
const LATIN_1: u8 = 1;
const HANGUL: u8 = 2;
const CJK: u8 = 3;
const ASCII_LETTERS: u8 = 4;

/// The two charsets `0x6c9a70` returns, the primary and the secondary; a name's first unit picks
/// whichever takes it, primary first, and every later unit must be of that one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NameCharsets {
    primary: u8,
    secondary: u8,
}

impl NameCharsets {
    /// What `Wow.ini`'s `Region` and the locale index give. A `Region` of 1 to 5 (signed,
    /// `0x6c9a78`-`0x6c9a80`; `0x402b7f` hands it over at start-up, `0x6c91b8` stores it) reads the
    /// pairs at `0x86c9e4`: 1 (1, 1), 2 (2, 4), 3 (1, 1), 4 (3, 1), 5 (3, 1). Any other reads the
    /// locale's: the table at `0x86ca0c` (1, 2, 1, 1, 3, 3, 1, then 0, which reads as index 0)
    /// for the primary, and 4 for koKR's secondary, 1 for every other (`0x6c9aa1`-`0x6c9acb`).
    pub fn for_client(region: i32, locale: usize) -> Self {
        const REGIONS: [(u8, u8); 5] = [(1, 1), (2, 4), (1, 1), (3, 1), (3, 1)];
        const LOCALES: [u8; 8] = [1, 2, 1, 1, 3, 3, 1, 0];
        const KO_KR: usize = 1;
        if let 1..=5 = region {
            let (primary, secondary) = REGIONS[(region - 1) as usize];
            return Self { primary, secondary };
        }
        // An index past the table or whose entry is 0 reads row 0.
        let index = match LOCALES.get(locale) {
            Some(&entry) if entry != 0 => locale,
            _ => 0,
        };
        Self {
            primary: LOCALES[index],
            secondary: if index == KO_KR {
                ASCII_LETTERS
            } else {
                LATIN_1
            },
        }
    }
}

impl Default for NameCharsets {
    /// enUS's, and any locale that reads the Latin charset twice.
    fn default() -> Self {
        Self::for_client(0, 0)
    }
}

/// `0x5abd50`: the status of `name` (`CHAR_NAME_NO_NAME` 0x45 to `CHAR_NAME_SUCCESS` 0x52). `None`
/// is a null name.
pub fn check_char_name(name: Option<&str>, lists: &dyn NameLists, charsets: NameCharsets) -> u8 {
    // `0x6c9ae0`: no extra characters, every list, the charset's cap.
    let all = Searched {
        names_profanity: true,
        chat_profanity: true,
        names_reserved: true,
    };
    let v = validate(name, &[], lists, all, charsets);
    let cap = match v.charset {
        LATIN_1 => LATIN_MAX_UNITS,
        CJK => CJK_MAX_UNITS,
        _ => OTHER_MAX_UNITS,
    };
    0x45 + if v.code == SUCCESS && v.units > cap {
        TOO_LONG
    } else {
        v.code
    }
}

/// The status of a full character name: two names, each valid by [`check_char_name`], separated by
/// one ASCII space (`CHAR_NAME_NO_NAME` 0x45 to `CHAR_NAME_SUCCESS` 0x52).
///
/// Deviation: 1.12.1 has no full character names; this two-part policy is benilla's own, for the
/// later names feature. Each part keeps the single-name charset, length, repetition and list rules,
/// so the cap and triple test apply per part, while all three lists are also searched over the
/// spaced whole and the parts concatenated, so restricted content spanning the boundary is caught.
/// The input is neither trimmed nor normalized, and exactly one ASCII space is required.
pub fn check_full_char_name(
    name: Option<&str>,
    lists: &dyn NameLists,
    charsets: NameCharsets,
) -> u8 {
    let Some(name) = name else {
        return 0x45 + NO_NAME;
    };
    if name.is_empty() {
        return 0x45 + NO_NAME;
    }
    let Some((first, last)) = name.split_once(' ') else {
        return 0x45 + INVALID_CHARACTER;
    };
    // Each part is exactly the existing single-name check: its order and cap are preserved, and a
    // first-part failure is returned before the last part is inspected.
    let status = check_char_name(Some(first), lists, charsets);
    if status != CHAR_NAME_SUCCESS {
        return status;
    }
    let status = check_char_name(Some(last), lists, charsets);
    if status != CHAR_NAME_SUCCESS {
        return status;
    }
    // The combined subjects are word-list checks only; running the single-name walk or cap over
    // either would reject valid full names such as two 12-unit parts or a boundary triple.
    let joined = format!("{first}{last}");
    let restricted = |subject: &str| {
        lists.names_profanity(subject)
            || lists.chat_profanity(subject)
            || lists.names_reserved(subject)
    };
    if restricted(name) || restricted(&joined) {
        return 0x45 + RESERVED;
    }
    CHAR_NAME_SUCCESS
}

/// Prepares a full character name for storage: validates the entered spelling, capitalizes each
/// part, then validates the canonical spelling again. Returns the canonical `"First Last"` string.
///
/// Both checks use the same lists and charsets and both include the per-part checks and the
/// combined restricted-content checks. The raw spelling is checked first, so its error status is
/// returned unchanged even when capitalization would fail later. No charset is widened and no
/// alternate spelling is attempted: for example, a leading `ß` capitalizes to the single character
/// `ẞ`, which the existing charset rejects, so the result is `Err(0x48)`.
pub fn prepare_full_char_name(
    name: Option<&str>,
    lists: &dyn NameLists,
    charsets: NameCharsets,
) -> Result<String, u8> {
    let status = check_full_char_name(name, lists, charsets);
    if status != CHAR_NAME_SUCCESS {
        return Err(status);
    }

    // A successful full-name check guarantees a present, nonempty, exactly-two-part string.
    let name = name.ok_or(0x45 + NO_NAME)?;
    let (first, last) = name.split_once(' ').ok_or(0x45 + INVALID_CHARACTER)?;

    let canonical = format!(
        "{} {}",
        capitalize_name_part(first),
        capitalize_name_part(last)
    );

    let status = check_full_char_name(Some(&canonical), lists, charsets);
    if status != CHAR_NAME_SUCCESS {
        return Err(status);
    }

    Ok(canonical)
}

/// Capitalizes one validated name part: the first scalar uppercase, the rest lowercase. The
/// existing client alphabet admits lowercase ß, þ and ÿ but not their uppercase counterparts, so
/// those initials deliberately become the single characters ẞ, Þ and Ÿ and are rejected by the
/// canonical validation. No multi-character expansion or fallback is performed.
fn capitalize_name_part(part: &str) -> String {
    let mut chars = part.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };

    let first = if first == 'ß' {
        'ẞ'
    } else {
        first.to_uppercase().next().unwrap_or(first)
    };

    let mut capitalized = String::with_capacity(part.len());
    capitalized.push(first);
    capitalized.extend(chars.flat_map(|c| c.to_lowercase()));
    capitalized
}

/// `0x6c9b70`: the validator's code for guild name `name`, [`SUCCESS`] for one it accepts.
pub(crate) fn check_guild_name(name: &str, lists: &dyn NameLists, charsets: NameCharsets) -> u8 {
    let profanity = Searched {
        names_profanity: true,
        chat_profanity: true,
        names_reserved: false,
    };
    let v = validate(Some(name), &[SPACE], lists, profanity, charsets);
    if v.code == SUCCESS && v.units > GUILD_MAX_UNITS {
        TOO_LONG
    } else {
        v.code
    }
}

/// `0x6c96c0` over `name`, taking the units of `extras` outside the charset and searching the
/// lists `searched` turns on.
fn validate(
    name: Option<&str>,
    extras: &[u16],
    lists: &dyn NameLists,
    searched: Searched,
    charsets: NameCharsets,
) -> Verdict {
    let mut v = Verdict {
        code: NO_NAME,
        units: 0,
        charset: 0,
    };
    let Some(name) = name else {
        return v;
    };
    let units: Vec<u16> = name.encode_utf16().collect();
    if units.len() >= UNITS {
        v.code = FAILURE;
        return v;
    }
    let fold = |u: u16| -> u16 {
        if (0x41..=0x5a).contains(&u) {
            u + 0x20
        } else {
            u
        }
    };
    // The last apostrophe's and the last space's index (`[ebp+0xc]`, `[ebp-0xc]`, -1 before).
    let mut apostrophe: Option<usize> = None;
    let mut space: Option<usize> = None;
    for (i, &unit) in units.iter().enumerate() {
        v.units = i + 1;
        if v.charset == 0 {
            v.charset = [charsets.primary, charsets.secondary]
                .into_iter()
                .find(|&set| in_charset(set, unit))
                .unwrap_or(0);
        }
        if v.charset != 0 && in_charset(v.charset, unit) {
            if i >= 2 {
                let (a, b) = (units[i - 2], units[i - 1]);
                let same = if v.charset == LATIN_1 {
                    fold(a) == fold(b) && fold(a) == fold(unit)
                } else {
                    a == b && a == unit
                };
                if same {
                    v.code = THREE_CONSECUTIVE;
                    return v;
                }
            }
            continue;
        }
        if !extras.contains(&unit) {
            v.code = INVALID_CHARACTER;
            return v;
        }
        let refused = match unit {
            APOSTROPHE if i == 0 => Some(INVALID_APOSTROPHE),
            APOSTROPHE if apostrophe.is_some() => Some(MULTIPLE_APOSTROPHES),
            APOSTROPHE => {
                apostrophe = Some(i);
                None
            }
            SPACE if i == 0 => Some(INVALID_SPACE),
            SPACE if space == Some(i - 1) => Some(CONSECUTIVE_SPACES),
            SPACE => {
                space = Some(i);
                None
            }
            _ => None,
        };
        if let Some(code) = refused {
            v.code = code;
            return v;
        }
    }
    let last = units.len().checked_sub(1);
    v.code = if units.len() < 2 {
        // 0 or 1 by the count (`0x6c98ac setne`).
        if units.is_empty() {
            NO_NAME
        } else {
            TOO_SHORT
        }
    } else if apostrophe == last {
        INVALID_APOSTROPHE
    } else if space == last {
        INVALID_SPACE
    } else if (searched.names_profanity && lists.names_profanity(name))
        || (searched.chat_profanity && lists.chat_profanity(name))
        || (searched.names_reserved && lists.names_reserved(name))
    {
        RESERVED
    } else {
        SUCCESS
    };
    v
}

/// Whether `unit` is in charset `set` (`0x6c9dd0`); an id it does not know, 0 included, takes none.
fn in_charset(set: u8, unit: u16) -> bool {
    match set {
        LATIN_1 => in_latin_1(unit),
        HANGUL => hangul::is_listed(unit),
        CJK => in_cjk(unit),
        ASCII_LETTERS => matches!(unit, 0x41..=0x5a | 0x61..=0x7a),
        _ => false,
    }
}

/// Charset 1 (`0x6c9c60`): A-Z, a-z, and `0xC0`-`0xFF` but `0xD7`, `0xDE` and `0xF7`.
fn in_latin_1(u: u16) -> bool {
    matches!(u, 0x41..=0x5a | 0x61..=0x7a | 0xc0..=0xff) && !matches!(u, 0xd7 | 0xde | 0xf7)
}

/// Charset 3 (`0x6c9d80`, through `0x6c9dc0`): the CJK ideographs, U+4E00-U+9FFF, the extension A
/// block U+3400-U+4DFF and the compatibility ideographs U+F900-U+FAFF.
fn in_cjk(u: u16) -> bool {
    matches!(u, 0x3400..=0x4dff | 0x4e00..=0x9fff | 0xf900..=0xfaff)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A list that reserves any name holding one of its words, ASCII-folded.
    #[derive(Clone)]
    struct Words(&'static [&'static str]);

    impl NameLists for Words {
        fn names_profanity(&self, _: &str) -> bool {
            false
        }

        fn chat_profanity(&self, _: &str) -> bool {
            false
        }

        fn names_reserved(&self, name: &str) -> bool {
            let name = name.to_ascii_lowercase();
            self.0.iter().any(|w| name.contains(w))
        }
    }

    fn check(name: &str) -> u8 {
        check_char_name(
            Some(name),
            &Words(&["blizz"]),
            NameCharsets::for_client(0, 0),
        )
    }

    /// The status of `name` on a client of this `Region` and locale index.
    fn check_on(region: i32, locale: usize, name: &str) -> u8 {
        check_char_name(
            Some(name),
            &Words(&["blizz"]),
            NameCharsets::for_client(region, locale),
        )
    }

    /// One name per code the check produces.
    #[test]
    fn each_code_the_check_produces() {
        assert_eq!(
            check_char_name(None, &NoNameLists, NameCharsets::default()),
            0x45,
            "NO_NAME, a null name"
        );
        assert_eq!(check(""), 0x45, "NO_NAME");
        assert_eq!(check("A"), 0x46, "TOO_SHORT");
        assert_eq!(check("Abcdefghijklm"), 0x47, "TOO_LONG, 13 units");
        assert_eq!(check("Abc1"), 0x48, "INVALID_CHARACTER");
        assert_eq!(check("Ab cd"), 0x48, "a space is outside the charset");
        assert_eq!(check("Ab'cd"), 0x48, "an apostrophe is outside the charset");
        assert_eq!(check("Blizzbob"), 0x4b, "RESERVED");
        assert_eq!(check("Abbbc"), 0x4e, "THREE_CONSECUTIVE");
        assert_eq!(check(&"a".repeat(0x400)), 0x51, "FAILURE, 1024 units");
        assert_eq!(check("Ab"), 0x52, "SUCCESS");
        assert_eq!(check("Abcdefghijkl"), 0x52, "12 units is the cap");
    }

    /// The charset's edges, its Latin-1 half included, and its three holes.
    #[test]
    fn charset_1_is_the_letters_and_latin_1_less_three() {
        for ok in ["Zoë", "Ægir", "Øyvind", "Ðaß", "Ab\u{ff}", "Ab\u{c0}"] {
            assert_eq!(check(ok), 0x52, "{ok}");
        }
        for bad in [
            "Ab\u{d7}",
            "Ab\u{de}",
            "Ab\u{f7}",
            "Ab\u{bf}",
            "Ab\u{100}",
            "Ab_",
            "Abé1",
        ] {
            assert_eq!(check(bad), 0x48, "{bad:?}");
        }
    }

    /// Three in a row folds ASCII case (`0x40f92b`), and only ASCII: `Éé` is two units apart.
    #[test]
    fn three_in_a_row_folds_ascii_only() {
        assert_eq!(check("AaAb"), 0x4e);
        assert_eq!(check("Bbb"), 0x4e);
        assert_eq!(check("ÉéÉb"), 0x52);
        assert_eq!(check("ÉÉÉb"), 0x4e);
        assert_eq!(check("Aabb"), 0x52, "two pairs are not three");
    }

    /// The order, where two rules fail one name: the walk decides by position between an invalid
    /// unit and a triple, ahead of the short test, the lists and the length.
    #[test]
    fn the_order_of_the_checks() {
        // The walk, by position.
        assert_eq!(check("Aaa1"), 0x4e, "the triple ends before the digit");
        assert_eq!(check("A1aaa"), 0x48, "the digit comes before the triple");
        // The walk before the short test: one bad unit is INVALID, not TOO_SHORT.
        assert_eq!(check("1"), 0x48);
        // The walk before the lists, the lists before the length.
        assert_eq!(check("Blizz1"), 0x48);
        assert_eq!(check("Blizzz"), 0x4e);
        assert_eq!(
            check("Blizzardbobby"),
            0x4b,
            "13 units and reserved is RESERVED"
        );
        // The length last of all: a 13-unit name with a digit is INVALID.
        assert_eq!(check("Abcdefghijkl1"), 0x48);
        assert_eq!(check("Abcdefghijkkk"), 0x4e);
        // The conversion first: a name too long to convert fails whatever it holds.
        assert_eq!(check(&"1".repeat(0x400)), 0x51);
        assert_eq!(
            check(&"1".repeat(0x3ff)),
            0x48,
            "1023 units and the NUL fit"
        );
        // A surrogate pair is two units and outside the charset.
        assert_eq!(check("Ab\u{1f600}"), 0x48);
    }

    /// `0x6c9a70` over the bytes' tables: a `Region` of 1 to 5 picks its pair, any other value
    /// reads the locale's, and the Latin pair (1, 1) is every locale but koKR, zhCN and zhTW.
    #[test]
    fn the_pair_is_the_regions_else_the_locales() {
        let pair = |region, locale| {
            let c = NameCharsets::for_client(region, locale);
            (c.primary, c.secondary)
        };
        for (region, want) in [
            (1, (1, 1)),
            (2, (2, 4)),
            (3, (1, 1)),
            (4, (3, 1)),
            (5, (3, 1)),
        ] {
            for locale in 0..8 {
                assert_eq!(
                    pair(region, locale),
                    want,
                    "Region {region}, locale {locale}"
                );
            }
        }
        // locale index -> (primary, secondary): enUS, koKR, frFR, deDE, zhCN, zhTW, esES, xxYY.
        let by_locale = [
            (1, 1),
            (2, 4),
            (1, 1),
            (1, 1),
            (3, 1),
            (3, 1),
            (1, 1),
            (1, 1),
        ];
        for (locale, want) in by_locale.into_iter().enumerate() {
            for region in [0, 6, -1, 1000] {
                assert_eq!(
                    pair(region, locale),
                    want,
                    "Region {region}, locale {locale}"
                );
            }
        }
        // An index past the table reads row 0, as the client's does.
        assert_eq!(pair(0, 9), (1, 1));
        assert_eq!(NameCharsets::default(), NameCharsets::for_client(0, 0));
        // The extremes of the signed word are no region, not an overflow.
        assert_eq!(pair(i32::MIN, 1), (2, 4));
        assert_eq!(pair(i32::MAX, 4), (3, 1));
    }

    /// koKR: a Hangul name passes, and the same bytes fail on every Latin locale. The first
    /// unit picks the charset: a name opening in Latin letters is held to ASCII letters.
    #[test]
    fn a_hangul_name_passes_on_korean_and_fails_on_a_latin_locale() {
        for name in ["한글", "가나다라", "김철수"] {
            assert_eq!(check_on(0, 1, name), 0x52, "{name} on koKR");
            for locale in [0, 2, 3, 6] {
                assert_eq!(check_on(0, locale, name), 0x48, "{name} on locale {locale}");
            }
        }
        // The first unit opens the secondary (ASCII letters), so the accents of charset 1 and
        // Hangul after it are outside.
        assert_eq!(check_on(0, 1, "Abc"), 0x52);
        assert_eq!(check_on(0, 1, "Zoë"), 0x48);
        assert_eq!(check_on(0, 1, "A한"), 0x48);
        // Opening in Hangul, a Latin letter after it is outside the name's charset.
        assert_eq!(check_on(0, 1, "한a"), 0x48);
        // Only the 2,350 listed syllables: 갂 (U+AC02) is in the block but not the list.
        assert_eq!(check_on(0, 1, "\u{ac02}\u{ac00}"), 0x48);
        assert_eq!(check_on(0, 1, "\u{ac00}\u{ac01}"), 0x52);
        // The decomposed jamo are not the syllables.
        assert_eq!(check_on(0, 1, "\u{1112}\u{1161}"), 0x48);
    }

    /// zhCN and zhTW: ideographs pass, six at most, and Latin letters after the first
    /// ideograph are outside; a name opening in ASCII letters falls to the secondary, charset 1.
    #[test]
    fn a_chinese_name_passes_on_chinese_locales_up_to_six() {
        for locale in [4, 5] {
            assert_eq!(check_on(0, locale, "李小龙"), 0x52, "locale {locale}");
            assert_eq!(check_on(0, locale, "歐陽修"), 0x52);
            assert_eq!(check_on(0, locale, "一二三四五六"), 0x52, "six is the cap");
            assert_eq!(
                check_on(0, locale, "一二三四五六七"),
                0x47,
                "seven is TOO_LONG"
            );
            assert_eq!(check_on(0, locale, "李a"), 0x48);
            assert_eq!(
                check_on(0, locale, "Zoë"),
                0x52,
                "the secondary is charset 1"
            );
            assert_eq!(
                check_on(0, locale, "Abcdefghijkl"),
                0x52,
                "charset 1 keeps 12"
            );
            assert_eq!(check_on(0, locale, "한글"), 0x48);
        }
        // Extension A and the compatibility block are ideographs too.
        assert_eq!(check_on(0, 4, "\u{3400}\u{f900}"), 0x52);
        // The enUS client has none of this.
        assert_eq!(check_on(0, 0, "李小龙"), 0x48);
    }

    /// A `Region` overrides the locale: Region 2 is koKR's pair on an enUS install, Region 4 and 5
    /// zhCN's, and Region 1 or 3 Latin on a koKR install.
    #[test]
    fn the_region_overrides_the_locale() {
        assert_eq!(check_on(2, 0, "한글"), 0x52);
        assert_eq!(check_on(4, 0, "李小龙"), 0x52);
        assert_eq!(check_on(5, 0, "李小龙"), 0x52);
        for region in [1, 3] {
            assert_eq!(check_on(region, 1, "한글"), 0x48, "Region {region}");
            assert_eq!(check_on(region, 1, "Zoë"), 0x52, "Region {region}");
        }
        assert_eq!(
            check_on(6, 1, "한글"),
            0x52,
            "Region 6 is none: the locale's"
        );
    }

    /// The cap follows the chosen charset: 12 for charset 1, 8 for 2 and 4, 6 for 3.
    #[test]
    fn the_cap_follows_the_chosen_charset() {
        let hangul = |n: usize| "가".repeat(n);
        // The same syllable repeated trips the triple test, so vary it over listed syllables.
        let varied =
            |n: usize| ["가", "나", "다", "라", "마", "바", "사", "아", "자", "차"][..n].concat();
        assert_eq!(check_on(0, 1, &varied(8)), 0x52, "eight Hangul");
        assert_eq!(check_on(0, 1, &varied(9)), 0x47, "nine Hangul");
        // koKR's secondary, ASCII letters: also 8.
        assert_eq!(check_on(0, 1, "Abcdefgh"), 0x52);
        assert_eq!(
            check_on(0, 1, "Abcdefghi"),
            0x47,
            "charset 4 caps at 8, not 12"
        );
        // Latin on enUS stays 12.
        assert_eq!(check_on(0, 0, "Abcdefghijkl"), 0x52);
        assert_eq!(check_on(0, 0, "Abcdefghijklm"), 0x47);
        // The cap is last: three of one syllable is the triple, whatever the length.
        assert_eq!(check_on(0, 1, &hangul(3)), 0x4e);
    }

    /// The triple test folds ASCII case for charset 1 alone; every other charset compares exactly.
    #[test]
    fn the_triple_test_is_exact_outside_charset_1() {
        // Charset 1 on enUS: Aaa is three of one letter.
        assert_eq!(check_on(0, 0, "Aaab"), 0x4e);
        // koKR's secondary is charset 4, whose triple is exact: Aaa is not three alike.
        assert_eq!(check_on(0, 1, "Aaab"), 0x52);
        assert_eq!(check_on(0, 1, "aaab"), 0x4e);
        // Hangul and ideographs likewise.
        assert_eq!(check_on(0, 1, "가가가나"), 0x4e);
        assert_eq!(check_on(0, 1, "가가나가"), 0x52);
        assert_eq!(check_on(0, 4, "一一一二"), 0x4e);
        assert_eq!(check_on(0, 4, "一一二一"), 0x52);
    }

    /// The first offending unit decides across charsets as within one: a bad second unit is
    /// INVALID before the triple or the cap, and the lists come before the cap.
    #[test]
    fn the_order_holds_across_charsets() {
        assert_eq!(check_on(0, 1, "한1글"), 0x48);
        assert_eq!(check_on(0, 1, "가"), 0x46, "one syllable is TOO_SHORT");
        assert_eq!(check_on(0, 1, "1"), 0x48);
        assert_eq!(check_on(0, 1, "가나다라마바사아자1"), 0x48);
    }

    /// The extra characters' own rules (`0x6c9833`-`0x6c9872`, `0x6c98e0`-`0x6c98f9`), with both
    /// the apostrophe and the space passed: neither may open or close the name, an apostrophe
    /// comes once, a space never twice in a row, and any other unit outside the charset is
    /// invalid. The walk decides by position, before the short test and the lists.
    #[test]
    fn the_extra_characters_rules() {
        let code = |name: &str| {
            let all = Searched {
                names_profanity: true,
                chat_profanity: true,
                names_reserved: true,
            };
            validate(
                Some(name),
                &[APOSTROPHE, SPACE],
                &Words(&["blizz"]),
                all,
                NameCharsets::default(),
            )
            .code
        };
        for (name, want) in [
            ("'Ab", INVALID_APOSTROPHE),
            ("Ab'", INVALID_APOSTROPHE),
            ("A'b'c", MULTIPLE_APOSTROPHES),
            ("A''b", MULTIPLE_APOSTROPHES),
            ("A'b", SUCCESS),
            (" Ab", INVALID_SPACE),
            ("Ab ", INVALID_SPACE),
            ("A  b", CONSECUTIVE_SPACES),
            ("A b c", SUCCESS),
            ("A-b", INVALID_CHARACTER),
            ("'", INVALID_APOSTROPHE),
            (" ", INVALID_SPACE),
            ("Blizz b", RESERVED),
            ("A1  b", INVALID_CHARACTER),
            ("Aaa b", THREE_CONSECUTIVE),
        ] {
            assert_eq!(code(name), want, "{name:?}");
        }
    }

    /// `0x6c9b70`: a space allowed, the two profanity lists searched and the reserved one not,
    /// and 24 units whatever the charset.
    #[test]
    fn the_guild_check_caps_at_24_and_skips_the_reserved_list() {
        let guild =
            |name: &str| check_guild_name(name, &Words(&["blizz"]), NameCharsets::default());
        assert_eq!(
            guild("Blizzard Fans"),
            SUCCESS,
            "Words fills NamesReserved alone"
        );
        assert_eq!(guild("Abcdefghij Abcdefghij Ab"), SUCCESS, "24 units");
        assert_eq!(guild("Abcdefghij Abcdefghij Abc"), TOO_LONG, "25 units");
        assert_eq!(guild("Ab cd"), SUCCESS);
        assert_eq!(check("Ab cd"), 0x48, "the character check takes no space");
    }

    /// One word list per predicate, each match ASCII-folded and unanchored, so a test can put a
    /// restriction in exactly one list.
    #[derive(Clone, Copy)]
    struct Lists {
        names_profanity: &'static [&'static str],
        chat_profanity: &'static [&'static str],
        names_reserved: &'static [&'static str],
    }

    impl Lists {
        const NONE: Self = Self {
            names_profanity: &[],
            chat_profanity: &[],
            names_reserved: &[],
        };

        fn any(words: &[&str], name: &str) -> bool {
            let name = name.to_ascii_lowercase();
            words.iter().any(|word| name.contains(word))
        }
    }

    impl NameLists for Lists {
        fn names_profanity(&self, name: &str) -> bool {
            Self::any(self.names_profanity, name)
        }

        fn chat_profanity(&self, name: &str) -> bool {
            Self::any(self.chat_profanity, name)
        }

        fn names_reserved(&self, name: &str) -> bool {
            Self::any(self.names_reserved, name)
        }
    }

    fn check_full(name: &str) -> u8 {
        check_full_char_name(Some(name), &NoNameLists, NameCharsets::default())
    }

    fn check_full_with(name: &str, lists: &dyn NameLists) -> u8 {
        check_full_char_name(Some(name), lists, NameCharsets::default())
    }

    fn check_full_on(region: i32, locale: usize, name: &str) -> u8 {
        check_full_char_name(
            Some(name),
            &NoNameLists,
            NameCharsets::for_client(region, locale),
        )
    }

    /// Two valid parts, each held to the single-name rules, with the Latin cap applied per part
    /// rather than to the full string. No capitalization is imposed.
    #[test]
    fn full_name_accepts_two_valid_parts_and_maximum_latin_length() {
        assert_eq!(check_full("Asha Brightvale"), 0x52, "two valid parts");
        assert_eq!(
            check_full("aSHA brightVALE"),
            0x52,
            "validated, not capitalized"
        );
        assert_eq!(
            check_full("Abcdefghijkl Mnopqrstuvwx"),
            0x52,
            "two 12-unit parts, 25 units with the space"
        );
        assert_eq!(
            check_full("Abcdefghijklm Brightvale"),
            0x47,
            "13-unit first part"
        );
        assert_eq!(check_full("Asha Abcdefghijklm"), 0x47, "13-unit last part");
    }

    /// The exact two-part shape: one ASCII space, no empty part, no extra component, and no
    /// repair of tabs, newlines, NBSP, embedded NULs or doubled spaces.
    #[test]
    fn full_name_rejects_missing_parts_and_malformed_separators() {
        assert_eq!(
            check_full_char_name(None, &NoNameLists, NameCharsets::default()),
            0x45,
            "None"
        );
        assert_eq!(check_full(""), 0x45, "empty");
        assert_eq!(check_full("Asha"), 0x48, "no separator");
        assert_eq!(check_full("Asha "), 0x45, "empty last part");
        assert_eq!(check_full(" Brightvale"), 0x45, "empty first part");
        assert_eq!(check_full("Asha  Brightvale"), 0x48, "doubled space");
        assert_eq!(check_full("Asha Bright vale"), 0x48, "third component");
        assert_eq!(check_full("Asha Brightvale Extra"), 0x48, "four components");
        for separator in ["\t", "\n", "\u{a0}"] {
            assert_eq!(
                check_full(&format!("Asha{separator}Brightvale")),
                0x48,
                "{separator:?} is not the separator"
            );
        }
        assert_eq!(
            check_full("As\u{0}ha Brightvale"),
            0x48,
            "NUL in the first part"
        );
        assert_eq!(
            check_full("Asha Bright\u{0}vale"),
            0x48,
            "NUL in the last part"
        );
        // Malformed inputs are answered with an error, not repaired into a valid pair.
        assert_ne!(check_full("Asha "), 0x52);
        assert_ne!(check_full("Asha\tBrightvale"), 0x52);
        assert_ne!(check_full("Asha  Brightvale"), 0x52);
    }

    /// Each part keeps the existing single-name codes and their order, and a first-part failure
    /// is returned before the last part is checked.
    #[test]
    fn full_name_preserves_per_part_errors_and_order() {
        let reserved = Words(&["blizz"]);
        assert_eq!(check_full("A Brightvale"), 0x46, "short first part");
        assert_eq!(check_full("Asha B"), 0x46, "short last part");
        assert_eq!(check_full("As1a Brightvale"), 0x48, "invalid first part");
        assert_eq!(check_full("Asha Bright1vale"), 0x48, "invalid last part");
        assert_eq!(check_full("Aaab Brightvale"), 0x4e, "triple in first part");
        assert_eq!(check_full("Asha Bbbright"), 0x4e, "triple in last part");
        assert_eq!(
            check_full_with("Blizz Brightvale", &reserved),
            0x4b,
            "reserved first part"
        );
        assert_eq!(
            check_full_with("Asha Blizz", &reserved),
            0x4b,
            "reserved last part"
        );
        // Both parts fail, differently: the first part's status wins.
        assert_eq!(
            check_full_with("A Blizz", &reserved),
            0x46,
            "first part's TOO_SHORT precedes the last part's RESERVED"
        );
        assert_eq!(
            check_full_with("Blizz A", &reserved),
            0x4b,
            "first part's RESERVED precedes the last part's TOO_SHORT"
        );
        assert_eq!(
            check_full_with("A1 Blizz", &reserved),
            0x48,
            "first part's INVALID_CHARACTER precedes the last part's RESERVED"
        );
        // The existing within-part order still decides: the walk by position, then the lists,
        // then the length cap.
        assert_eq!(check_full("Aaab1 Brightvale"), 0x4e, "triple before digit");
        assert_eq!(check_full("A1aaa Brightvale"), 0x48, "digit before triple");
        assert_eq!(
            check_full_with("Blizzardbobby Brightvale", &reserved),
            0x4b,
            "RESERVED before the 13-unit cap"
        );
    }

    /// Every list sees both combined subjects: the name with its space and the two parts joined
    /// without it. The individual parts pass, so only the boundary checks can reject these.
    #[test]
    fn full_name_checks_each_list_across_the_boundary() {
        let spaced_profanity = Lists {
            names_profanity: &["asha bright"],
            ..Lists::NONE
        };
        let spaced_chat = Lists {
            chat_profanity: &["asha bright"],
            ..Lists::NONE
        };
        let spaced_reserved = Lists {
            names_reserved: &["asha bright"],
            ..Lists::NONE
        };
        for lists in [&spaced_profanity, &spaced_chat, &spaced_reserved] {
            assert_eq!(
                check_full_with("Asha Brightvale", lists),
                0x4b,
                "each list rejects the spaced form"
            );
            for part in ["Asha", "Brightvale"] {
                assert_eq!(
                    check_char_name(Some(part), lists, NameCharsets::default()),
                    0x52,
                    "{part} passes on its own"
                );
            }
        }

        let joined_profanity = Lists {
            names_profanity: &["ashabright"],
            ..Lists::NONE
        };
        let joined_chat = Lists {
            chat_profanity: &["ashabright"],
            ..Lists::NONE
        };
        let joined_reserved = Lists {
            names_reserved: &["ashabright"],
            ..Lists::NONE
        };
        for lists in [&joined_profanity, &joined_chat, &joined_reserved] {
            assert_eq!(
                check_full_with("Asha Brightvale", lists),
                0x4b,
                "each list rejects the concatenated form"
            );
            for part in ["Asha", "Brightvale"] {
                assert_eq!(
                    check_char_name(Some(part), lists, NameCharsets::default()),
                    0x52,
                    "{part} passes on its own"
                );
            }
        }

        // "Bl Izza" passes each part; "BlIzza" folded contains "blizz".
        let restricted = Lists {
            names_reserved: &["blizz"],
            ..Lists::NONE
        };
        assert_eq!(check("Bl"), 0x52);
        assert_eq!(check("Izza"), 0x52);
        assert_eq!(
            check_full_with("Bl Izza", &restricted),
            0x4b,
            "case-folded restriction across the boundary"
        );
        assert_eq!(
            check_full_with("Asha Brightvale", &restricted),
            0x52,
            "valid nonmatching control"
        );
    }

    /// The single-name cap and triple test apply inside each part only: a valid combined length
    /// and a triple formed only at the boundary are not rejected.
    #[test]
    fn full_name_keeps_length_and_repetition_checks_per_part() {
        assert_eq!(
            check_full("Abcdefghijkl Mnopqrstuvwx"),
            0x52,
            "two 12-unit parts, 25 units together"
        );
        assert_eq!(
            check_full("Abb Bc"),
            0x52,
            "concatenation is AbbBc: b, b, B is not a per-part triple"
        );
        assert_eq!(
            check_full("Ab Bbc"),
            0x52,
            "concatenation is AbBbc: b, B, b is not a per-part triple"
        );
        assert_eq!(check_full("Abbb Bc"), 0x4e, "triple inside the first part");
        assert_eq!(check_full("Abb Bccc"), 0x4e, "triple inside the last part");
        assert_eq!(
            check_full("Aaab Bc"),
            0x4e,
            "case-folded triple inside the first part"
        );
    }

    /// Each part chooses its charset and cap under the same [`NameCharsets`]; one name may mix
    /// the primary in one part and the secondary in the other.
    #[test]
    fn full_name_keeps_locale_caps_per_part() {
        assert_eq!(
            check_full_on(0, 0, "가나다라마바사아 Abcdefgh"),
            0x48,
            "enUS rejects Hangul"
        );
        assert_eq!(
            check_full_on(0, 1, "가나다라마바사아 Abcdefgh"),
            0x52,
            "koKR: eight Hangul then eight ASCII"
        );
        assert_eq!(
            check_full_on(0, 1, "가나다라마바사아자 Abcdefgh"),
            0x47,
            "koKR: nine Hangul in the first part"
        );
        assert_eq!(
            check_full_on(0, 1, "가나다라마바사아 Abcdefghi"),
            0x47,
            "koKR: nine ASCII in the last part"
        );
        assert_eq!(
            check_full_on(0, 1, "Abcdefgh 가나다라마바사아자"),
            0x47,
            "koKR: nine Hangul in the last part"
        );
        assert_eq!(
            check_full_on(0, 4, "一二三四五六 李小龙歐陽修"),
            0x52,
            "zhCN: six ideographs in each part"
        );
        assert_eq!(
            check_full_on(0, 4, "一二三四五六七 李小龙歐陽修"),
            0x47,
            "zhCN: seven ideographs in the first part"
        );
        assert_eq!(
            check_full_on(0, 4, "一二三四五六 李小龙歐陽修文"),
            0x47,
            "zhCN: seven ideographs in the last part"
        );
        // The parts pick different charsets from the same pair.
        assert_eq!(
            check_full_on(0, 1, "Abcdefgh 가나다라"),
            0x52,
            "koKR: ASCII (secondary) then Hangul (primary)"
        );
        assert_eq!(
            check_full_on(0, 4, "Abcdefgh 李小龙"),
            0x52,
            "zhCN: ASCII (secondary) then ideographs (primary)"
        );
    }

    fn prepare(name: &str) -> Result<String, u8> {
        prepare_full_char_name(Some(name), &NoNameLists, NameCharsets::default())
    }

    fn prepare_with(name: &str, lists: &dyn NameLists) -> Result<String, u8> {
        prepare_full_char_name(Some(name), lists, NameCharsets::default())
    }

    fn prepare_on(region: i32, locale: usize, name: &str) -> Result<String, u8> {
        prepare_full_char_name(
            Some(name),
            &NoNameLists,
            NameCharsets::for_client(region, locale),
        )
    }

    fn assert_ok(result: Result<String, u8>, expected: &str) {
        match result {
            Ok(value) => assert_eq!(value, expected),
            Err(status) => panic!("expected Ok({expected:?}), got Err(0x{status:02x})"),
        }
    }

    fn assert_err(result: Result<String, u8>, expected: u8) {
        match result {
            Ok(value) => panic!("expected Err(0x{expected:02x}), got Ok({value:?})"),
            Err(status) => assert_eq!(status, expected),
        }
    }

    /// The canonical spelling is produced only after the raw spelling passes.
    #[test]
    fn prepare_full_name_capitalizes_each_part() {
        assert_ok(prepare("aSHA brightVALE"), "Asha Brightvale");
        assert_ok(prepare("éclair bravo"), "Éclair Bravo");
        assert_ok(prepare("Asha Brightvale"), "Asha Brightvale");
        assert_ok(prepare("Éclair Bravo"), "Éclair Bravo");
    }

    /// The existing alphabet has no uppercase ẞ, Þ, or Ÿ. Raw lowercase is
    /// accepted, but the capitalized form is rejected rather than widened or
    /// replaced with an alternate spelling.
    #[test]
    fn prepare_full_name_rejects_unrepresentable_initials() {
        for (raw, capitalized) in [
            ("ßeta Brightvale", "ẞeta Brightvale"),
            ("þor Brightvale", "Þor Brightvale"),
            ("ÿard Brightvale", "Ÿard Brightvale"),
            ("Asha ßeta", "Asha ẞeta"),
            ("Asha þor", "Asha Þor"),
            ("Asha ÿard", "Asha Ÿard"),
        ] {
            assert_eq!(check_full(raw), 0x52, "raw spelling is admitted");
            assert_eq!(check_full(capitalized), 0x48, "capitalized is not");
            assert_err(prepare(raw), 0x48);
        }
        assert_err(prepare("ẞeta Brightvale"), 0x48);
        assert_err(prepare("Þor Brightvale"), 0x48);
        assert_err(prepare("Ÿard Brightvale"), 0x48);
    }

    /// Lowercase exceptional letters remain legal away from the initial. Only
    /// the canonical initial mapping is rejected.
    #[test]
    fn prepare_full_name_preserves_allowed_internal_letters() {
        assert_ok(prepare("aßa aþa"), "Aßa Aþa");
        assert_ok(prepare("aÿa brightvale"), "Aÿa Brightvale");
    }

    /// Raw failures are returned unchanged and take precedence over anything
    /// capitalization would discover later.
    #[test]
    fn prepare_full_name_forwards_raw_errors() {
        assert_err(
            prepare_full_char_name(None, &NoNameLists, NameCharsets::default()),
            0x45,
        );
        assert_err(prepare(""), 0x45);
        assert_err(prepare("Asha "), 0x45);
        assert_err(prepare("Asha"), 0x48);
        assert_err(prepare("As1a Brightvale"), 0x48);
        assert_err(prepare("A Brightvale"), 0x46);
        assert_err(prepare("Abcdefghijklm Brightvale"), 0x47);
        assert_err(prepare("Aaab Brightvale"), 0x4e);
        assert_err(prepare_with("Blizz Brightvale", &Words(&["blizz"])), 0x4b);
        assert_err(prepare("ßeta A"), 0x46);
    }

    /// Capitalization can create a repetition the raw spelling did not have;
    /// the canonical string is checked again.
    #[test]
    fn prepare_full_name_rechecks_repetition_after_casing() {
        assert_eq!(check_full("éÉéÉ Vale"), 0x52);
        assert_err(prepare("éÉéÉ Vale"), 0x4e);
        assert_ok(prepare("éÉé Vale"), "Ééé Vale");
    }

    /// The same three lists are checked against the canonical spaced and
    /// concatenated forms, even when the raw forms pass.
    #[test]
    fn prepare_full_name_rechecks_combined_lists_after_casing() {
        let spaced_names = Lists {
            names_profanity: &["Éclair brightvale"],
            ..Lists::NONE
        };
        let spaced_chat = Lists {
            chat_profanity: &["Éclair brightvale"],
            ..Lists::NONE
        };
        let spaced_reserved = Lists {
            names_reserved: &["Éclair brightvale"],
            ..Lists::NONE
        };
        for lists in [&spaced_names, &spaced_chat, &spaced_reserved] {
            assert_eq!(check_full_with("éclair brightVALE", lists), 0x52);
            assert_eq!(
                check_char_name(Some("éclair"), lists, NameCharsets::default()),
                0x52
            );
            assert_eq!(
                check_char_name(Some("brightVALE"), lists, NameCharsets::default()),
                0x52
            );
            assert_err(prepare_with("éclair brightVALE", lists), 0x4b);
        }

        let joined_names = Lists {
            names_profanity: &["Éclairbrightvale"],
            ..Lists::NONE
        };
        let joined_chat = Lists {
            chat_profanity: &["Éclairbrightvale"],
            ..Lists::NONE
        };
        let joined_reserved = Lists {
            names_reserved: &["Éclairbrightvale"],
            ..Lists::NONE
        };
        for lists in [&joined_names, &joined_chat, &joined_reserved] {
            assert_eq!(check_full_with("éclair brightVALE", lists), 0x52);
            assert_eq!(
                check_char_name(Some("éclair"), lists, NameCharsets::default()),
                0x52
            );
            assert_eq!(
                check_char_name(Some("brightVALE"), lists, NameCharsets::default()),
                0x52
            );
            assert_err(prepare_with("éclair brightVALE", lists), 0x4b);
        }

        assert_ok(
            prepare_with("éclair brightVALE", &Lists::NONE),
            "Éclair Brightvale",
        );
    }

    /// Raw and canonical checks both retain the existing charset and length
    /// limits; canonicalization does not bypass them.
    #[test]
    fn prepare_full_name_keeps_locale_caps_and_maximum_lengths() {
        assert_ok(
            prepare("Abcdefghijkl Mnopqrstuvwx"),
            "Abcdefghijkl Mnopqrstuvwx",
        );
        assert_err(prepare("Abcdefghijklm Brightvale"), 0x47);
        assert_err(prepare("Asha Abcdefghijklm"), 0x47);

        assert_ok(
            prepare_on(0, 1, "가나다라마바사아 Abcdefgh"),
            "가나다라마바사아 Abcdefgh",
        );
        assert_err(prepare_on(0, 1, "가나다라마바사아자 Abcdefgh"), 0x47);
        assert_err(prepare_on(0, 1, "가나다라마바사아 Abcdefghi"), 0x47);
        assert_err(prepare_on(0, 1, "Abcdefgh 가나다라마바사아자"), 0x47);

        assert_ok(
            prepare_on(0, 4, "一二三四五六 李小龙歐陽修"),
            "一二三四五六 李小龙歐陽修",
        );
        assert_err(prepare_on(0, 4, "一二三四五六七 李小龙歐陽修"), 0x47);
        assert_err(prepare_on(0, 4, "一二三四五六 李小龙歐陽修文"), 0x47);
        assert_ok(
            prepare_on(0, 4, "Abcdefghijkl 李小龙"),
            "Abcdefghijkl 李小龙",
        );
        assert_err(prepare_on(0, 4, "Abcdefghijklm 李小龙"), 0x47);
        assert_err(prepare_on(0, 0, "가나다라마바사아 Abcdefgh"), 0x48);
    }
}
