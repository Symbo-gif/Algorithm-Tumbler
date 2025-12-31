# Copyright 2025 Damien Davison, Michael Maillet, and Sacha Davison, Recursive AI Devs
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from algorithms.library.manager import ComponentLibrary
from algorithms.core.types import INT, LIST


def test_seed_library_contains_algorithms():
    lib = ComponentLibrary.seed_default()
    assert len(lib.components) >= 3

    sorters = lib.query_by_spec("sort")
    assert sorters
    sorted_result = sorters[0].instantiate().execute([[3, 1, 2]])
    assert sorted_result == [1, 2, 3]


def test_library_save_and_load_round_trip(tmp_path):
    lib = ComponentLibrary.seed_default()
    output_path = tmp_path / "library.json"
    lib.save(output_path)

    loaded = ComponentLibrary.load(output_path)
    assert len(loaded.components) == len(lib.components)

    factorial = loaded.get("factorial").instantiate()
    assert factorial.execute([5]) == 120

    int_list_components = loaded.query_by_type(LIST(INT))
    assert any(comp.spec for comp in int_list_components)
