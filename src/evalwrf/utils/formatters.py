def millions_formatter(x, pos):
    return f"{x * 1e-6:.1f}M"


def megawatt_formatter(x, pos):
    return f"{x * 1e-6:.1f} MW"


def gigawatt_formatter(x, pos):
    return f"{x * 1e-9:.1f} GW"
