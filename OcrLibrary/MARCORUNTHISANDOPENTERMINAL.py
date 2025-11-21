# totally_normal_script.py

def _build_centered_line(width: int, codes):
    text = "".join(chr(c) for c in codes)
    pad = max(0, width - len(text))
    left = pad // 2
    right = pad - left
    return " " * left + text + " " * right


def _build_banner():
    width = 40

    # "L I M P   B I Z K I T" encoded as ASCII codes
    title_codes = [
        76, 32, 73, 32, 77, 32, 80, 32, 32, 32,
        66, 32, 73, 32, 90, 32, 75, 32, 73, 32, 84
    ]

    top_bottom = "=" * width
    middle = _build_centered_line(width, title_codes)

    return [top_bottom, middle, top_bottom]


def _build_rock_hand_block():
    # \m/
    hand_codes = [92, 109, 47]
    # |
    #  \
    wrist_codes = [124, 92]
    # / \
    arm_codes = [47, 32, 92]

    hand = "".join(chr(c) for c in hand_codes)
    wrist = "".join(chr(c) for c in wrist_codes)
    arm = "".join(chr(c) for c in arm_codes)

    # ⚡ (lightning) = 9889
    lightning = "".join(chr(c) for c in [9889, 32, 9889])

    lines = []
    lines.append(" " * 14 + hand + " " * 8 + lightning)
    lines.append(" " * 15 + wrist)
    lines.append(" " * 15 + arm)
    return lines


def main():
    lines = []
    lines.extend(_build_banner())
    lines.append("")
    lines.extend(_build_rock_hand_block())
    print("\n".join(lines))


if __name__ == "__main__":
    main()
