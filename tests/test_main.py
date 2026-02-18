from src.main import main


def test_main_prints_greeting(capsys) -> None:
    exit_code = main(["--name", "Part 1"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip() == "Hello from Part 1!"
