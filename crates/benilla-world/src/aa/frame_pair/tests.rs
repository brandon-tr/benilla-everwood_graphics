use super::*;

fn pair(generation: u64) -> TemporalFramePair<u8> {
    TemporalFramePair::new(generation, [10, 20])
}

#[test]
fn previous_for_write_requires_exact_predecessor_without_current() {
    let masks = pair(7);
    assert_eq!(masks.previous_for_write(0, 7), None);
    assert_eq!(masks.previous_for_write(1, 7), None);
    assert!(masks.encode_current(1, 7, |_| true));
    assert_eq!(masks.current(2, 7), None);
    assert_eq!(masks.previous_for_write(2, 7), Some(&20));
    assert_eq!(masks.previous_for_write(2, 8), None);
    assert_eq!(masks.previous_for_write(3, 7), None);

    let previous = masks.previous_for_write(2, 7).copied();
    let mut destinations = Vec::new();
    assert!(masks.encode_current(2, 7, |slot| {
        destinations.push((previous, *slot));
        true
    }));
    assert_eq!(destinations, vec![(Some(20), 10)]);
    assert_eq!(masks.current(2, 7), Some(&10));
}

#[test]
fn first_encoding_publishes_current_without_previous() {
    let masks = pair(7);
    assert!(masks.encode_current(1, 7, |slot| {
        assert_eq!(*slot, 20);
        true
    }));
    assert_eq!(masks.current(1, 7), Some(&20));
    assert_eq!(masks.previous(1, 7), None);
}

#[test]
fn consecutive_encodings_preserve_distinct_payloads() {
    let masks = pair(7);
    let destination = std::cell::RefCell::new(Vec::new());
    assert!(masks.encode_current(1, 7, |slot| {
        destination.borrow_mut().push(*slot);
        true
    }));
    assert_eq!(masks.current(1, 7), Some(&20));
    assert_eq!(masks.previous(1, 7), None);

    assert!(masks.encode_current(2, 7, |slot| {
        destination.borrow_mut().push(*slot);
        assert_eq!(*slot, 10);
        true
    }));
    assert_eq!(*destination.borrow(), vec![20, 10]);
    assert_eq!(masks.current(2, 7), Some(&10));
    assert_eq!(masks.previous(2, 7), Some(&20));

    assert!(masks.encode_current(3, 7, |slot| {
        destination.borrow_mut().push(*slot);
        assert_eq!(*slot, 20);
        true
    }));
    assert_eq!(*destination.borrow(), vec![20, 10, 20]);
    assert_eq!(masks.current(3, 7), Some(&20));
    assert_eq!(masks.previous(3, 7), Some(&10));
    assert_ne!(masks.current(3, 7), masks.previous(3, 7));
}

#[test]
fn skipped_epoch_cannot_supply_previous() {
    let masks = pair(7);
    assert!(masks.encode_current(1, 7, |_| true));
    assert!(masks.encode_current(3, 7, |_| true));
    assert_eq!(masks.current(3, 7), Some(&20));
    assert_eq!(masks.previous(3, 7), None);

    assert!(masks.encode_current(4, 7, |_| true));
    assert_eq!(masks.current(4, 7), Some(&10));
    assert_eq!(masks.previous(4, 7), Some(&20));
}

#[test]
fn failed_encoding_does_not_publish() {
    let masks = pair(7);
    assert!(masks.encode_current(1, 7, |_| true));
    assert!(!masks.encode_current(2, 7, |_| false));
    assert_eq!(masks.current(2, 7), None);
    assert_eq!(masks.previous(2, 7), None);
    assert_eq!(masks.current(1, 7), Some(&20));
    assert_eq!(masks.previous(1, 7), None);
}

#[test]
fn zero_and_wrong_generation_do_not_invoke_encoder() {
    let masks = pair(7);
    let calls = std::cell::Cell::new(0);
    assert!(!masks.encode_current(0, 7, |_| {
        calls.set(calls.get() + 1);
        true
    }));
    assert!(!masks.encode_current(1, 8, |_| {
        calls.set(calls.get() + 1);
        true
    }));
    assert_eq!(calls.get(), 0);
    assert_eq!(masks.current(1, 7), None);
}

#[test]
fn generation_replacement_starts_empty() {
    let old = pair(7);
    assert!(old.encode_current(1, 7, |_| true));
    assert_eq!(old.current(1, 7), Some(&20));

    let replacement = pair(8);
    assert_eq!(replacement.current(1, 7), None);
    assert_eq!(replacement.previous(1, 7), None);
    assert_eq!(replacement.current(1, 8), None);
    assert_eq!(replacement.previous(1, 8), None);
}

#[test]
fn independent_pairs_do_not_share_publication() {
    let first = pair(7);
    let second = pair(7);
    assert!(first.encode_current(1, 7, |_| true));
    assert_eq!(first.current(1, 7), Some(&20));
    assert_eq!(second.current(1, 7), None);
    assert_eq!(second.previous(1, 7), None);
}

#[test]
fn epoch_wrap_does_not_create_consecutive_history() {
    let masks = pair(7);
    assert!(masks.encode_current(u64::MAX, 7, |_| true));
    assert_eq!(masks.current(u64::MAX, 7), Some(&20));
    assert!(masks.encode_current(1, 7, |_| true));
    assert_eq!(masks.current(1, 7), Some(&20));
    assert_eq!(masks.previous(1, 7), None);
}

#[test]
fn current_must_encode_before_previous_is_advertised() {
    let masks = pair(7);
    assert_eq!(masks.previous(1, 7), None);
    assert!(masks.encode_current(1, 7, |_| true));
    assert_eq!(masks.previous(1, 7), None);
    assert_eq!(masks.current(1, 7), Some(&20));
    assert_eq!(masks.previous(2, 7), None);
}
