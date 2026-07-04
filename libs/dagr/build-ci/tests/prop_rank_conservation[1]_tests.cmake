add_test([=[RankConservation.InvariantHoldsAcrossAllOperations]=]  /workspace/helm-project/libs/dagr/build-ci/tests/prop_rank_conservation [==[--gtest_filter=RankConservation.InvariantHoldsAcrossAllOperations]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[RankConservation.InvariantHoldsAcrossAllOperations]=]  PROPERTIES WORKING_DIRECTORY /workspace/helm-project/libs/dagr/build-ci/tests SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==] ENVIRONMENT [==[RC_PARAMS=max_success=1000]==] TIMEOUT 120)
set(  prop_rank_conservation_TESTS RankConservation.InvariantHoldsAcrossAllOperations)
