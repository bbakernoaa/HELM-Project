add_test([=[AcyclicityDetection.CyclicGraphThrows]=]  /workspace/helm-project/libs/dagr/build-ci/tests/prop_acyclicity_detection [==[--gtest_filter=AcyclicityDetection.CyclicGraphThrows]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[AcyclicityDetection.CyclicGraphThrows]=]  PROPERTIES WORKING_DIRECTORY /workspace/helm-project/libs/dagr/build-ci/tests SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==] ENVIRONMENT [==[RC_PARAMS=max_success=1000]==] TIMEOUT 120)
set(  prop_acyclicity_detection_TESTS AcyclicityDetection.CyclicGraphThrows)
