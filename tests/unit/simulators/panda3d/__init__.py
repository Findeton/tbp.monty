# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

try:
    import panda3d  # noqa: F401
except ImportError:
    raise unittest.SkipTest("Panda3D optional dependency not installed.")
