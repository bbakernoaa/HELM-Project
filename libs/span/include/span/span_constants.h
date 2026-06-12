/* SPDX-License-Identifier: Apache-2.0 */
/* SPAN Constants — C-compatible header for Fortran iso_c_binding */

#ifndef HELM_SPAN_CONSTANTS_H
#define HELM_SPAN_CONSTANTS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Memory space tokens */
#define HELM_SPAN_MEM_HOST         0
#define HELM_SPAN_MEM_CUDA_DEVICE  1
#define HELM_SPAN_MEM_HIP_DEVICE   2

/* Error codes */
#define HELM_SPAN_SUCCESS                    0
#define HELM_SPAN_ERR_NULL_PTR              -1
#define HELM_SPAN_ERR_INVALID_RANK          -2
#define HELM_SPAN_ERR_INVALID_HANDLE        -3
#define HELM_SPAN_ERR_INVALID_MEMORY_SPACE  -4
#define HELM_SPAN_ERR_REGISTRY_FULL         -5
#define HELM_SPAN_ERR_INTERNAL             -99

/* Maximum supported array rank */
#define HELM_SPAN_MAX_RANK  7

#ifdef __cplusplus
}
#endif

#endif /* HELM_SPAN_CONSTANTS_H */
