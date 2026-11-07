"""
Shared pytest setup.

Tests draw figures but never show them, so use matplotlib's non-interactive Agg
backend. The macOS default, `macosx`, starts a GUI app on the first figure, which
puts a Python icon in the Dock for the length of the run. This runs before any
test module imports pyplot, which is what makes it take effect.
"""

import matplotlib

matplotlib.use("Agg")
