add_test([=[FailureCascade.TransitiveCancellationOnFailure]=]  /workspace/helm-project/libs/dagr/build-ci/tests/prop_failure_cascade [==[--gtest_filter=FailureCascade.TransitiveCancellationOnFailure]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[FailureCascade.TransitiveCancellationOnFailure]=]  PROPERTIES WORKING_DIRECTORY /workspace/helm-project/libs/dagr/build-ci/tests SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==] ENVIRONMENT [==[RC_PARAMS=max_success=1000]==] TIMEOUT 120)
set(  prop_failure_cascade_TESTS FailureCascade.TransitiveCancellationOnFailure)
