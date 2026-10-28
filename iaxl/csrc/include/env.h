// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdlib.h>
#include <strings.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

static inline int env_int(const char *name, int fallback) {
    const char *v = getenv(name);
    if (!v || !*v)
        return fallback;
    int x = atoi(v);
    return x > 0 ? x : fallback;
}

static inline int env_nonnegative_int(const char *name, int fallback) {
    const char *v = getenv(name);
    if (!v || !*v)
        return fallback;
    int x = atoi(v);
    return x >= 0 ? x : fallback;
}

static inline const char *env_str(const char *name, const char *fallback) {
    const char *v = getenv(name);
    return (v && *v) ? v : fallback;
}

static inline int env_bool(const char *name, int fallback) {
    const char *v = getenv(name);
    if (!v || !*v)
        return fallback;
    if (!strcasecmp(v, "true") || !strcasecmp(v, "yes") || !strcasecmp(v, "on"))
        return 1;
    if (!strcasecmp(v, "false") || !strcasecmp(v, "no") || !strcasecmp(v, "off"))
        return 0;
    return atoi(v) != 0;
}

static inline double env_double(const char *name, double fallback) {
    const char *v = getenv(name);
    if (!v || !*v)
        return fallback;
    char *end = NULL;
    double x = strtod(v, &end);
    if (end == v)
        return fallback;
    return x;
}

static inline double env_watermark(const char *name, double fallback) {
    double wm = env_double(name, fallback);
    if (wm > 1.0 && wm <= 100.0)
        wm /= 100.0;
    if (wm <= 0.0 || wm > 1.0)
        return fallback;
    return wm;
}

struct Envs {

    int IAXL_ZIP_SRC_CAP;
    int IAXL_ZIP_DST_CAP;
    int IAXL_ZIP_LEVEL;
    int IAXL_ZIP_SEGMENT_SIZE_KB;

    bool IAXL_QAT_ZIP_ENABLE;
    bool IAXL_IAA_ZIP_ENABLE;
    bool IAXL_CPU_ZIP_ENABLE;
    int IAXL_QAT_INSTANCE_NUM;
    const char *(*IAXL_QAT_DEVICES)(void);
    int IAXL_QAT_ZIP_INSTANCES_PER_DEVICE;
    int IAXL_QAT_ZIP_QUEUE_DEPTH;
    int IAXL_IAA_INSTANCE_NUM;
    const char *(*IAXL_IAA_DEVICES)(void);
    int IAXL_IAA_ZIP_INSTANCES_PER_DEVICE;
    int IAXL_IAA_ZIP_QUEUE_DEPTH;
    int IAXL_CPU_ZIP_THREADS;
    int IAXL_OMP_THREAD_NUM;
    bool IAXL_USE_OMP;

    bool IAXL_KV_COMPRESSION;
    int IAXL_KV_LOSSY_TRUNC;
    bool IAXL_KV_DATA_SHUFFLE;
    bool IAXL_ZIP_EXPONENT_ONLY;
    int IAXL_CACHE_CACHEGROUP_SIZE;
    int IAXL_CACHE_CACHEGROUP_NUM;
    bool IAXL_CACHE_AUTO_SPILL_ENABLE;
    bool IAXL_CACHE_AUTO_SPILL_EVICT_ONLY;
    double IAXL_CACHE_HIGH_WATERMARK;
    double IAXL_CACHE_LOW_WATERMARK;
    int IAXL_CACHE_SPILL_BATCH_GROUPS;
    int IAXL_CACHE_SPILL_PACE_MS;
    int IAXL_CACHE_EVICT_GRACE_MS;

    bool IAXL_DSA_GD_ENABLE;
    bool IAXL_DSA_GD_RESET_ON_DESTROY;
    const char *(*IAXL_DSA_WQS)(void);

    bool IAXL_DEBUG_LOG;
    const char *IAXL_PROFILE_MODE;
};

#ifdef __cplusplus
extern "C" {
#endif

extern struct Envs envs;

void envs_init(void);

#ifdef __cplusplus
}
#endif
