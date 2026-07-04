add_test([=[NoOrphan.AllNodesReachTerminalWithinLongestPathPlusOne]=]  /workspace/helm-project/libs/dagr/build-ci/tests/prop_no_orphan [==[--gtest_filter=NoOrphan.AllNodesReachTerminalWithinLongestPathPlusOne]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[NoOrphan.AllNodesReachTerminalWithinLongestPathPlusOne]=]  PROPERTIES WORKING_DIRECTORY /workspace/helm-project/libs/dagr/build-ci/tests SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==] ENVIRONMENT [==[RC_PARAMS=max_success=1000]==] TIMEOUT 120)
set(  prop_no_orphan_TESTS NoOrphan.AllNodesReachTerminalWithinLongestPathPlusOne)
