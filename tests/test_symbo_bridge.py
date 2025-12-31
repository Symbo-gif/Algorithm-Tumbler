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

from algorithms.symbo_bridge import SymboBridge
from algorithms.analysis.semantic_summarizer import SemanticSummarizer


def test_symbo_bridge_simplify_and_status():
    bridge = SymboBridge(enable_llm=False)
    status = bridge.status()
    assert "llm_available" in status
    result = bridge.simplify_expression("1 + 2 * 3")
    assert str(result["result"]).startswith("7")


def test_symbo_bridge_story_reasoning():
    bridge = SymboBridge(enable_llm=False)
    analysis = bridge.reason_about_story("Alice has 3 apples and gets 2 more.")
    assert "entities" in analysis
    assert len(analysis["entities"]) >= 2


def test_semantic_summarizer_with_symbo_bridge():
    bridge = SymboBridge(enable_llm=False)
    summarizer = SemanticSummarizer(symbo_bridge=bridge)
    summary = summarizer.summarize("def foo():\n    return 42")
    assert summary.lower().startswith("symbo")
