from src.main import build_parser, main


def test_main_prints_greeting(capsys) -> None:
    exit_code = main(["--name", "Part 1"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip() == "Hello from Part 1!"


def test_main_parser_includes_fedawa_subcommand() -> None:
    args = build_parser().parse_args(["fedawa", "--config", "configs/fedawa.yaml"])

    assert args.command == "fedawa"


def test_main_parser_includes_dfl_subcommand() -> None:
    args = build_parser().parse_args(["dfl", "--config", "configs/dfl.yaml"])

    assert args.command == "dfl"
