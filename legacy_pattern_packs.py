"""Expose the legacy pattern pack subset in a human-readable way."""
import pattern_packs


def main():
    print("✅ Legacy pack subset is available via pattern_packs")
    s40_count = len(pattern_packs.S40)
    pack164950_count = len(pattern_packs.PACK_164950_NUMBERS)

    print(f"   S40 numbers           : {s40_count}")
    print(f"   164950 family numbers : {pack164950_count}")
    print(f"   3-digit families      : {len(pattern_packs.PACK_3_FAMILIES)}")
    print(f"   4-digit families      : {len(pattern_packs.PACK_4_FAMILIES)}")

    print("\nSample legacy tags:")
    samples = [0, 14, 55, 64, 98]
    for num in samples:
        tags = pattern_packs.get_digit_pack_tags(num)
        print(f" • {num:02d}: {tags}")


if __name__ == "__main__":
    main()
