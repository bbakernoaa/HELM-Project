add_test([=[YAMLRoundTrip.StreamDescriptorsPreserved]=]  /workspace/helm-project/libs/dagr/build-ci/tests/prop_yaml_roundtrip [==[--gtest_filter=YAMLRoundTrip.StreamDescriptorsPreserved]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[YAMLRoundTrip.StreamDescriptorsPreserved]=]  PROPERTIES WORKING_DIRECTORY /workspace/helm-project/libs/dagr/build-ci/tests SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==] ENVIRONMENT [==[RC_PARAMS=max_success=1000]==] TIMEOUT 120)
set(  prop_yaml_roundtrip_TESTS YAMLRoundTrip.StreamDescriptorsPreserved)
