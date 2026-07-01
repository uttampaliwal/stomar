from src.automation.daily_runner import DailyAutomationRunner


def test_daily_automation_runner_initializes():
    runner = DailyAutomationRunner(tickers=["RELIANCE.NS"], db_path=":memory:")
    assert runner.tickers == ["RELIANCE.NS"]
    assert runner.db_path == ":memory:"
