"""Запуск анализа термокосы T4 и кросс-корреляции (term_blocks + crosscorr)."""

from __future__ import annotations

import sys
from pathlib import Path

BLOCK_SPECS = (
    (1, "part1_setup_and_fields.py", "Данные, схемы термокос, поля T и N, изотермы"),
    (2, "part2_spectra.py", "Спектры (амплитуда, PSD, Garrett–Munk)"),
    (3, "part3_waves_stats.py", "Волны: статистика, гистограммы, кластеры, ДИ"),
    (4, "part4_short_window.py", "Короткое окно цуга ±1 ч (st4_fig12)"),
)

_MODE_BLOCKS = {
    1: (1,),
    2: (1, 2),
    3: (1, 2, 3),
    4: (1, 2, 4),
    5: (1, 2, 3, 4),
}

# Пункты 6–8: crosscorr_thermistor (подрежимы 1, 2, 3).
XCORR_MENU = {
    6: 1,
    7: 2,
    8: 3,
}


def _exec_block(block_path: Path, context: dict) -> None:
    code = block_path.read_text(encoding="utf-8")
    exec(compile(code, str(block_path), "exec"), context)


def _print_menu() -> None:
    print("\n" + "=" * 60)
    print("АНАЛИЗ ТЕРМОКОСЫ И КРОСС-КОРРЕЛЯЦИЯ (Term.py)")
    print("=" * 60)
    for num, _fname, desc in BLOCK_SPECS:
        print(f"  {num} - {desc}")
    print("  5 - Все блоки термокосы (1+2+3+4)")
    print("  6 - Кросс-корр.: T4, изотермы на разных глубинах")
    print("  7 - Кросс-корр.: T1–T4, одна глубина, цуг ±1 ч")
    print("  8 - Кросс-корр.: оба расчёта (6+7)")
    print("  0 - Выход")


def _parse_mode_arg() -> int | None:
    if len(sys.argv) < 2:
        return None
    raw = sys.argv[1].strip().lstrip("-")
    if raw in ("h", "help"):
        print("Использование: python Term.py [0-8]")
        print("  1–5 — блоки term_blocks; 6–8 — кросс-корреляция")
        sys.exit(0)
    if raw.isdigit() and raw in "012345678":
        return int(raw)
    print(f"  Неизвестный режим: {sys.argv[1]!r}. Допустимо 0–8.")
    sys.exit(1)


def _ask_mode() -> int:
    preset = _parse_mode_arg()
    if preset is not None:
        return preset
    _print_menu()
    while True:
        raw = input("Выберите пункт (0-8, Enter = 5): ").strip() or "5"
        if raw in "012345678":
            return int(raw)
        print("  Введите 0, 1, 2, 3, 4, 5, 6, 7 или 8.")


def _run_term_blocks(to_run: tuple[int, ...], base_dir: Path) -> None:
    blocks_dir = base_dir / "term_blocks"
    by_num = {n: (blocks_dir / fname, desc) for n, fname, desc in BLOCK_SPECS}
    context = {"__name__": "__main__", "__file__": str(base_dir / "st4-stat.py")}
    for num in to_run:
        path, desc = by_num[num]
        if not path.is_file():
            raise FileNotFoundError(f"Нет файла блока: {path}")
        print(f"\n--- Блок {num}: {desc} ---")
        _exec_block(path, context)


def main() -> None:
    mode = _ask_mode()
    if mode == 0:
        print("Выход.")
        return

    base_dir = Path(__file__).resolve().parent

    if mode in XCORR_MENU:
        print(f"\nРежим {mode}: кросс-корреляция")
        import crosscorr_thermistor as xc

        xc.run_crosscorr(XCORR_MENU[mode])
        print("\nГотово.")
        return

    to_run = _MODE_BLOCKS[mode]
    print(f"\nРежим {mode}: блоки {list(to_run)}")
    _run_term_blocks(to_run, base_dir)
    print("\nГотово.")


if __name__ == "__main__":
    main()
