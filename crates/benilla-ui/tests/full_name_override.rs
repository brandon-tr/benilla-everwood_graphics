//! Executes the shipped two-field full-name overlay against actual script controls.
//!
//! This is a fork-specific check: stock 1.12.1 has one name field, while this fork splits the
//! entered full name into first and last parts before the Rust binding admits it.

use benilla_ui::script::UiScript;

const XML: &str = include_str!("../../benilla-app/assets/ui/AccountLoginRealmlist.xml");

fn overlay_script() -> String {
    let marker = "BenillaFullName_SubmitCreation";
    let marker_at = XML.find(marker).expect("full-name overlay marker");
    let start = XML[..marker_at].rfind("<Script>").expect("script start") + "<Script>".len();
    let end = XML[marker_at..].find("</Script>").expect("script end") + marker_at;
    XML[start..end].to_string()
}

fn fixture() -> UiScript {
    let s = UiScript::new().unwrap();
    s.run(
        r#"
        _G = getfenv(0)
        GlueDialog_Show = function(which, text)
            DIALOG_WHICH = which
            DIALOG_TEXT = text
        end
        local originalCreateFrame = CreateFrame
        CreateFrame = function(kind, name, parent, template)
            -- The synthetic VM has no stock InputBoxTemplate; the four-argument call shape is
            -- preserved cut from the template while the returned control is real.
            return originalCreateFrame(kind, name, parent)
        end
        P = CreateFrame("Frame", "P")
        local originalFont = P.CreateFontString
        P.CreateFontString = function(self, name, layer)
            local font = originalFont(self, name, layer)
            font:SetFont("fixture.ttf", 10)
            return font
        end

        CharacterCreateNameEdit = CreateFrame("EditBox", "CharacterCreateNameEdit", P)
        CharacterCreateNameEdit:SetMaxLetters(12)
        CharacterCreateNameEdit:SetWidth(180)
        CharacterCreateNameEdit:SetHeight(20)
        CharCreateOkayButton = CreateFrame("Button", "CharCreateOkayButton", P)
        CharCreateOkayButton:SetScript("OnClick", function()
            CREATED = CharacterCreateNameEdit:GetText()
        end)

        CharacterRenameEditBox = CreateFrame("EditBox", "CharacterRenameEditBox", P)
        CharacterRenameEditBox:SetMaxLetters(12)
        CharacterRenameEditBox:SetWidth(180)
        CharacterRenameEditBox:SetHeight(20)
        CharacterRenameButton1 = CreateFrame("Button", "CharacterRenameButton1", P)
        CharacterRenameButton1:SetScript("OnClick", function()
            RENAMED = CharacterRenameEditBox:GetText()
        end)
    "#,
    )
    .unwrap();
    s.run(&overlay_script()).unwrap();
    s
}

#[test]
fn creation_button_sends_both_parts_once_without_mutating_fields() {
    let s = fixture();
    s.run(
        r#"
        CharacterCreateNameEdit:SetText("Alice")
        BenillaCharacterCreateLastNameEdit:SetText("Smith")
        CharCreateOkayButton:Click()
        CREATED_AGAIN = nil
        CharCreateOkayButton:Click()
        CREATED_AGAIN = CREATED
        CREATED_FIRST = CharacterCreateNameEdit:GetText()
        CREATED_LAST = BenillaCharacterCreateLastNameEdit:GetText()
    "#,
    )
    .unwrap();
    assert_eq!(s.eval::<String>("return CREATED").unwrap(), "Alice Smith");
    assert_eq!(
        s.eval::<String>("return CREATED_AGAIN").unwrap(),
        "Alice Smith"
    );
    assert_eq!(s.eval::<String>("return CREATED_FIRST").unwrap(), "Alice");
    assert_eq!(s.eval::<String>("return CREATED_LAST").unwrap(), "Smith");
}

#[test]
fn creation_enter_takes_the_same_path_as_the_button() {
    let s = fixture();
    s.run(
        r#"
        CharacterCreateNameEdit:SetText("Alice")
        BenillaCharacterCreateLastNameEdit:SetText("Smith")
        CharacterCreateNameEdit:GetScript("OnEnterPressed")()
    "#,
    )
    .unwrap();
    assert_eq!(s.eval::<String>("return CREATED").unwrap(), "Alice Smith");
}

#[test]
fn maximum_parts_are_not_truncated_by_the_first_edit_cap() {
    let s = fixture();
    s.run(
        r#"
        CharacterCreateNameEdit:SetText("Abcdefghijkl")
        BenillaCharacterCreateLastNameEdit:SetText("Mnopqrstuvwx")
        CharCreateOkayButton:Click()
    "#,
    )
    .unwrap();
    assert_eq!(
        s.eval::<String>("return CREATED").unwrap(),
        "Abcdefghijkl Mnopqrstuvwx"
    );
}

#[test]
fn empty_part_sends_nothing_and_names_the_missing_half() {
    let first_msg = "Enter your character's first name.";
    let last_msg = "Enter your character's last name.";
    for (first, last, text) in [
        ("", "Smith", first_msg),
        ("Alice", "", last_msg),
        ("  ", "Smith", first_msg),
        ("Alice", "  ", last_msg),
    ] {
        let s = fixture();
        s.run(&format!(
            r#"
            CharacterCreateNameEdit:SetText("{first}")
            BenillaCharacterCreateLastNameEdit:SetText("{last}")
            CharCreateOkayButton:Click()
            "#
        ))
        .unwrap();
        assert_eq!(
            s.eval::<Option<String>>("return CREATED").unwrap(),
            None,
            "{first:?}+{last:?}"
        );
        assert_eq!(
            s.eval::<(Option<String>, Option<String>)>("return DIALOG_WHICH, DIALOG_TEXT")
                .unwrap(),
            (Some("BENILLA_NAME_ERROR".into()), Some(text.into())),
            "{first:?}+{last:?}"
        );
    }
}

#[test]
fn rename_button_uses_its_own_pair_and_fields() {
    let s = fixture();
    s.run(
        r#"
        CharacterRenameEditBox:SetText("Alice")
        BenillaCharacterRenameLastNameEdit:SetText("Smith")
        CharacterRenameButton1:Click()
        RENAMED_FIRST = CharacterRenameEditBox:GetText()
        RENAMED_LAST = BenillaCharacterRenameLastNameEdit:GetText()
    "#,
    )
    .unwrap();
    assert_eq!(s.eval::<String>("return RENAMED").unwrap(), "Alice Smith");
    assert_eq!(s.eval::<String>("return RENAMED_FIRST").unwrap(), "Alice");
    assert_eq!(s.eval::<String>("return RENAMED_LAST").unwrap(), "Smith");
}

#[test]
fn reloading_the_overlay_does_not_duplicate_controls() {
    let s = fixture();
    let before = s.eval::<f64>("return P:GetNumChildren()").unwrap();
    s.run(&overlay_script()).unwrap();
    let after = s.eval::<f64>("return P:GetNumChildren()").unwrap();
    assert_eq!(before, after);
}

#[test]
fn rename_repeats_maximum_and_non_ascii_without_mutation() {
    let s = fixture();
    s.run(
        r#"
        CharacterRenameEditBox:SetText("Abcdefghijkl")
        BenillaCharacterRenameLastNameEdit:SetText("Mnopqrstuvwx")
        CharacterRenameButton1:Click()
        RENAME_MAX = RENAMED
        CharacterRenameButton1:Click()
        RENAME_REPEAT = RENAMED
        CharacterRenameEditBox:SetText("Éowyn")
        BenillaCharacterRenameLastNameEdit:SetText("Éclair")
        CharacterRenameButton1:Click()
        RENAME_UNICODE = RENAMED
        RENAME_FIRST = CharacterRenameEditBox:GetText()
        RENAME_LAST = BenillaCharacterRenameLastNameEdit:GetText()
    "#,
    )
    .unwrap();
    assert_eq!(
        s.eval::<String>("return RENAME_MAX").unwrap(),
        "Abcdefghijkl Mnopqrstuvwx"
    );
    assert_eq!(
        s.eval::<String>("return RENAME_REPEAT").unwrap(),
        "Abcdefghijkl Mnopqrstuvwx"
    );
    assert_eq!(
        s.eval::<String>("return RENAME_UNICODE").unwrap(),
        "Éowyn Éclair"
    );
    assert_eq!(s.eval::<String>("return RENAME_FIRST").unwrap(), "Éowyn");
    assert_eq!(s.eval::<String>("return RENAME_LAST").unwrap(), "Éclair");
}

#[test]
fn empty_rename_part_sends_nothing_and_names_the_missing_half() {
    let first_msg = "Enter your character's first name.";
    let last_msg = "Enter your character's last name.";
    for (first, last, text) in [
        ("", "Smith", first_msg),
        ("Alice", "", last_msg),
        ("  ", "Smith", first_msg),
        ("Alice", "  ", last_msg),
    ] {
        let s = fixture();
        s.run(&format!(
            r#"
            CharacterRenameEditBox:SetText("{first}")
            BenillaCharacterRenameLastNameEdit:SetText("{last}")
            CharacterRenameButton1:Click()
            "#
        ))
        .unwrap();
        assert_eq!(
            s.eval::<Option<String>>("return RENAMED").unwrap(),
            None,
            "{first:?}+{last:?}"
        );
        assert_eq!(
            s.eval::<(Option<String>, Option<String>)>("return DIALOG_WHICH, DIALOG_TEXT")
                .unwrap(),
            (Some("BENILLA_NAME_ERROR".into()), Some(text.into())),
            "{first:?}+{last:?}"
        );
    }
}

#[test]
fn labels_are_distinct_and_stable_across_reload() {
    let s = fixture();
    let labels = s
        .eval::<(String, String, String, String)>(
            r#"
        return BenillaCharacterCreateFirstNameLabel:GetText(),
               BenillaCharacterCreateLastNameLabel:GetText(),
               BenillaCharacterRenameFirstNameLabel:GetText(),
               BenillaCharacterRenameLastNameLabel:GetText()
        "#,
        )
        .unwrap();
    assert_eq!(
        labels,
        (
            "First name".to_string(),
            "Last name".to_string(),
            "First name".to_string(),
            "Last name".to_string()
        )
    );
    s.run(&overlay_script()).unwrap();
    let name = s
        .eval::<String>("return BenillaCharacterCreateFirstNameLabel:GetName()")
        .unwrap();
    assert_eq!(name, "BenillaCharacterCreateFirstNameLabel");
}

#[test]
fn rename_enter_and_tab_match_the_creation_flow() {
    let mut s = fixture();
    s.run(
        r#"P:Show()
           CharacterRenameEditBox:Show()
           BenillaCharacterRenameLastNameEdit:Show()
           CharacterRenameEditBox:SetText("Alice")
           BenillaCharacterRenameLastNameEdit:SetText("Smith")
           CharacterRenameEditBox:GetScript("OnEnterPressed")()"#,
    )
    .unwrap();
    assert_eq!(s.eval::<String>("return RENAMED").unwrap(), "Alice Smith");
    assert!(s.focus_editbox("CharacterRenameEditBox"));
    s.run(r#"CharacterRenameEditBox:GetScript("OnTabPressed")()"#)
        .unwrap();
    assert_eq!(
        s.focused_editbox_name().as_deref(),
        Some("BenillaCharacterRenameLastNameEdit")
    );
    s.run(r#"BenillaCharacterRenameLastNameEdit:GetScript("OnTabPressed")()"#)
        .unwrap();
    assert_eq!(
        s.focused_editbox_name().as_deref(),
        Some("CharacterRenameEditBox")
    );
}

#[test]
fn creation_keeps_supported_non_ascii_parts_whole() {
    let s = fixture();
    s.run(
        r#"
        CharacterCreateNameEdit:SetText("Éowyn")
        BenillaCharacterCreateLastNameEdit:SetText("Éclair")
        CharCreateOkayButton:Click()
        UNICODE_FIRST = CharacterCreateNameEdit:GetText()
        UNICODE_LAST = BenillaCharacterCreateLastNameEdit:GetText()
    "#,
    )
    .unwrap();
    assert_eq!(s.eval::<String>("return CREATED").unwrap(), "Éowyn Éclair");
    assert_eq!(s.eval::<String>("return UNICODE_FIRST").unwrap(), "Éowyn");
    assert_eq!(s.eval::<String>("return UNICODE_LAST").unwrap(), "Éclair");
}

#[test]
fn a_rejected_submission_keeps_both_parts() {
    let s = fixture();
    s.run(
        r#"
        CharCreateOkayButton.BenillaFullNameOriginal = function()
            REJECTED = true
        end
        CharacterCreateNameEdit:SetText("Alice")
        BenillaCharacterCreateLastNameEdit:SetText("Smith")
        CharCreateOkayButton:Click()
        REJECT_FIRST = CharacterCreateNameEdit:GetText()
        REJECT_LAST = BenillaCharacterCreateLastNameEdit:GetText()
    "#,
    )
    .unwrap();
    assert!(s.eval::<bool>("return REJECTED").unwrap());
    assert_eq!(s.eval::<String>("return REJECT_FIRST").unwrap(), "Alice");
    assert_eq!(s.eval::<String>("return REJECT_LAST").unwrap(), "Smith");
}

#[test]
fn tab_moves_focus_between_the_two_parts() {
    let mut s = fixture();
    s.run(
        r#"P:Show() CharacterCreateNameEdit:Show()
           BenillaCharacterCreateLastNameEdit:Show()"#,
    )
    .unwrap();
    assert!(s.focus_editbox("CharacterCreateNameEdit"));
    s.run(r#"CharacterCreateNameEdit:GetScript("OnTabPressed")()"#)
        .unwrap();
    assert_eq!(
        s.focused_editbox_name().as_deref(),
        Some("BenillaCharacterCreateLastNameEdit")
    );
    s.run(r#"BenillaCharacterCreateLastNameEdit:GetScript("OnTabPressed")()"#)
        .unwrap();
    assert_eq!(
        s.focused_editbox_name().as_deref(),
        Some("CharacterCreateNameEdit")
    );
}

#[test]
fn showing_the_rename_pair_clears_the_last_part_only() {
    let s = fixture();
    s.run(
        r#"
        CharacterRenameEditBox:SetText("Alice")
        BenillaCharacterRenameLastNameEdit:SetText("Smith")
        CharacterRenameEditBox:Hide()
        CharacterRenameEditBox:Show()
        RENAME_AFTER_FIRST = CharacterRenameEditBox:GetText()
        RENAME_AFTER_LAST = BenillaCharacterRenameLastNameEdit:GetText()
    "#,
    )
    .unwrap();
    assert_eq!(
        s.eval::<String>("return RENAME_AFTER_FIRST").unwrap(),
        "Alice"
    );
    assert_eq!(s.eval::<String>("return RENAME_AFTER_LAST").unwrap(), "");
}

#[test]
fn showing_the_creation_pair_clears_the_last_part_only() {
    let s = fixture();
    s.run(
        r#"
        CharacterCreateNameEdit:SetText("Alice")
        BenillaCharacterCreateLastNameEdit:SetText("Smith")
        CharacterCreateNameEdit:Hide()
        CharacterCreateNameEdit:Show()
        AFTER_FIRST = CharacterCreateNameEdit:GetText()
        AFTER_LAST = BenillaCharacterCreateLastNameEdit:GetText()
    "#,
    )
    .unwrap();
    assert_eq!(s.eval::<String>("return AFTER_FIRST").unwrap(), "Alice");
    assert_eq!(s.eval::<String>("return AFTER_LAST").unwrap(), "");
}
