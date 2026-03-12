def format_currency(value: float, prefix: str = "$") -> str:
    if value >= 1e9:
        return f"{prefix}{value/1e9:.1f}B"
    if value >= 1e6:
        return f"{prefix}{value/1e6:.1f}M"
    if value >= 1e3:
        return f"{prefix}{value/1e3:.1f}K"
    return f"{prefix}{value:.2f}"


def get_color(signal: str) -> str:
    signal = signal.upper()
    if any(x in signal for x in ["BULL", "UP", "LONG", "BUY"]):
        return "#00ff6a"
    if any(x in signal for x in ["BEAR", "DOWN", "SHORT", "SELL"]):
        return "#ff3355"
    return "#ffaa00"
