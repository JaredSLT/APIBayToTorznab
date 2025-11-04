// main.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#else
#include <sys/mman.h>
#endif

#if defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#endif

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define HAS_X86 1
#include <immintrin.h>
#ifndef _WIN32
#include <cpuid.h>
#endif
#endif

#if defined(__aarch64__) || (defined(__arm__) && defined(__ARM_NEON__))
#define HAS_ARM 1
#include <arm_neon.h>
#endif

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

#if defined(__powerpc__) || defined(__ppc__) || defined(__PPC__)
#define HAS_POWER 1
#include <altivec.h>
#endif

#if defined(__riscv) && defined(__riscv_vector)
#define HAS_RISCV 1
#include <riscv_vector.h>
#endif

#ifdef _MSC_VER
#define FORCE_INLINE __forceinline
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#else
#define FORCE_INLINE __attribute__((always_inline)) inline
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

struct Arena {
    char *base;
    char *cur;
    char *end;
};

typedef struct {
    char *data;
    size_t len;
} Str;

FORCE_INLINE void *arena_alloc(struct Arena *__restrict a, size_t sz) {
    size_t align = 16;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    align = 64;
#elif defined(__AVX2__) || defined(__ARM_FEATURE_SVE)
    align = 32;
#endif
    uintptr_t curr = (uintptr_t)a->cur;
    uintptr_t aligned = (curr + align - 1) & ~(align - 1);
    size_t pad = aligned - curr;
    a->cur += pad;
    if (UNLIKELY(a->cur + sz > a->end)) return NULL;
    char *p = a->cur;
    a->cur += sz;
    return p;
}

struct TorrentInfo {
    int64_t id;
    int64_t size;
    int category;
    int num_files;
    int seeders;
    int leechers;
    int anon;
    time_t added;
    char *info_hash;
    size_t info_hash_len;
    char *name;
    size_t name_len;
    char *status;
    size_t status_len;
    char *username;
    size_t username_len;
    char *imdb;
    size_t imdb_len;
};

static const char hex_digits[16] = "0123456789ABCDEF";
static const uint8_t is_url_safe[256] = {
    ['0'...'9'] = 1,
    ['A'...'Z'] = 1,
    ['a'...'z'] = 1,
    ['-'] = 1,
    ['_'] = 1,
    ['.'] = 1,
    ['~'] = 1,
};
static const uint8_t is_space_lut[256] = {
    [' '] = 1,
    ['\t'] = 1,
    ['\n'] = 1,
    ['\r'] = 1,
    ['\v'] = 1,
    ['\f'] = 1,
};
static const uint8_t is_digit_lut[256] = {
    ['0'...'9'] = 1,
};

typedef enum {
    CPU_SCALAR = -1,
    CPU_SSE2,
    CPU_SSSE3,
    CPU_AVX2,
    CPU_AVX512
} CpuLevel;

static CpuLevel cpu_level;

#if defined(HAS_ARM)
static bool has_sve = false;
#endif

#ifdef HAS_X86
void cpuid(int info[4], int leaf, int subleaf) {
#ifdef _MSC_VER
    __cpuidex(info, leaf, subleaf);
#else
    __cpuid_count(leaf, subleaf, info[0], info[1], info[2], info[3]);
#endif
}

CpuLevel get_cpu_level() {
    int info[4];
    cpuid(info, 1, 0);
    if (!(info[3] & (1 << 26))) return CPU_SCALAR;
    int ssse3 = (info[2] & (1 << 9)) ? 1 : 0;
    cpuid(info, 7, 0);
    int avx2 = (info[1] & (1 << 5)) ? 1 : 0;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    int avx512f = (info[1] & (1 << 16)) ? 1 : 0;
    int avx512bw = (info[1] & (1 << 30)) ? 1 : 0;
    int avx512vl = (info[1] & (1 << 31)) ? 1 : 0;
    if (avx512f && avx512bw && avx512vl) return CPU_AVX512;
#endif
    if (avx2) return CPU_AVX2;
    if (ssse3) return CPU_SSSE3;
    return CPU_SSE2;
}
#endif

// Scalar functions
void skip_whitespace_scalar(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos && is_space_lut[(unsigned char)*p]) p++;
    *pp = p;
}

void find_quote_scalar(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos && *p != '"') p++;
    *pp = p;
}

void skip_digits_scalar(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos && is_digit_lut[(unsigned char)*p]) p++;
    *pp = p;
}

void skip_post_object_scalar(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos && (is_space_lut[(unsigned char)*p] || *p == ',')) p++;
    *pp = p;
}

Str url_encode_scalar(const char *str, size_t len, struct Arena *__restrict a) {
    if (!str || len == 0) return (Str){NULL, 0};
    size_t buf_size = len * 3 + 1;
    char *buf = arena_alloc(a, buf_size);
    if (!buf) return (Str){NULL, 0};
    char *p = buf;
    const unsigned char *us = (const unsigned char *)str;
    const unsigned char *end = us + len;
#if defined(__clang__)
#pragma unroll 4
#elif defined(__GNUC__)
#pragma GCC unroll 4
#endif
    while (us < end) {
        unsigned char c = *us++;
        if (is_url_safe[c]) {
            *p++ = (char)c;
        } else {
            *p++ = '%';
            *p++ = hex_digits[c >> 4];
            *p++ = hex_digits[c & 0xF];
        }
    }
    return (Str){buf, (size_t)(p - buf)};
}

Str to_lower_scalar(const char *str, size_t len, struct Arena *__restrict a) {
    if (!str || len == 0) return (Str){NULL, 0};
    char *lower = arena_alloc(a, len + 1);
    if (!lower) return (Str){NULL, 0};
#if defined(__clang__)
#pragma unroll 4
#elif defined(__GNUC__)
#pragma GCC unroll 4
#endif
    for (size_t i = 0; i < len; i++) {
        char c = str[i];
        lower[i] = (c >= 'A' && c <= 'Z') ? c + 32 : c;
    }
    return (Str){lower, len};
}

static FORCE_INLINE char *mempcpy_inline(char *dst, const char *src, size_t n) {
    memcpy(dst, src, n);
    return dst + n;
}

#ifdef HAS_X86
void skip_whitespace_sse2(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 16) {
            while (p < end_pos && is_space_lut[(unsigned char)*p]) p++;
            break;
        }
        __m128i data = _mm_loadu_si128((const __m128i *)p);
        __m128i sp = _mm_cmpeq_epi8(data, _mm_set1_epi8(' '));
        __m128i tab = _mm_cmpeq_epi8(data, _mm_set1_epi8('\t'));
        __m128i nl = _mm_cmpeq_epi8(data, _mm_set1_epi8('\n'));
        __m128i cr = _mm_cmpeq_epi8(data, _mm_set1_epi8('\r'));
        __m128i vt = _mm_cmpeq_epi8(data, _mm_set1_epi8('\v'));
        __m128i ff = _mm_cmpeq_epi8(data, _mm_set1_epi8('\f'));
        __m128i is_ws = _mm_or_si128(sp, tab);
        is_ws = _mm_or_si128(is_ws, nl);
        is_ws = _mm_or_si128(is_ws, cr);
        is_ws = _mm_or_si128(is_ws, vt);
        is_ws = _mm_or_si128(is_ws, ff);
        int mask = _mm_movemask_epi8(is_ws);
        if (mask == 0xFFFF) {
            p += 16;
        } else {
            p += __builtin_ctz((unsigned int)(~mask));
            break;
        }
    }
    *pp = p;
}

void skip_whitespace_avx2(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 32) {
            skip_whitespace_sse2(&p, end_pos);
            break;
        }
        __m256i data = _mm256_loadu_si256((const __m256i *)p);
        __m256i sp = _mm256_cmpeq_epi8(data, _mm256_set1_epi8(' '));
        __m256i tab = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('\t'));
        __m256i nl = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('\n'));
        __m256i cr = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('\r'));
        __m256i vt = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('\v'));
        __m256i ff = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('\f'));
        __m256i is_ws = _mm256_or_si256(sp, tab);
        is_ws = _mm256_or_si256(is_ws, nl);
        is_ws = _mm256_or_si256(is_ws, cr);
        is_ws = _mm256_or_si256(is_ws, vt);
        is_ws = _mm256_or_si256(is_ws, ff);
        uint32_t mask = (uint32_t)_mm256_movemask_epi8(is_ws);
        if (mask == 0xFFFFFFFFU) {
            p += 32;
        } else {
            p += __builtin_ctz((unsigned int)(~mask));
            break;
        }
    }
    *pp = p;
}

void find_quote_sse2(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 16) {
            while (p < end_pos && *p != '"') p++;
            break;
        }
        __m128i data = _mm_loadu_si128((const __m128i *)p);
        __m128i quotes = _mm_cmpeq_epi8(data, _mm_set1_epi8('"'));
        int mask = _mm_movemask_epi8(quotes);
        if (mask != 0) {
            p += __builtin_ctz((unsigned int)mask);
            break;
        }
        p += 16;
    }
    *pp = p;
}

void find_quote_avx2(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 32) {
            find_quote_sse2(&p, end_pos);
            break;
        }
        __m256i data = _mm256_loadu_si256((const __m256i *)p);
        __m256i quotes = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('"'));
        uint32_t mask = (uint32_t)_mm256_movemask_epi8(quotes);
        if (mask != 0) {
            p += __builtin_ctz((unsigned int)mask);
            break;
        }
        p += 32;
    }
    *pp = p;
}

void skip_digits_sse2(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 16) {
            while (p < end_pos && is_digit_lut[(unsigned char)*p]) p++;
            break;
        }
        __m128i data = _mm_loadu_si128((const __m128i *)p);
        __m128i ge0 = _mm_cmpgt_epi8(data, _mm_set1_epi8('0' - 1));
        __m128i le9 = _mm_cmplt_epi8(data, _mm_set1_epi8('9' + 1));
        __m128i is_dig = _mm_and_si128(ge0, le9);
        int mask = _mm_movemask_epi8(is_dig);
        if (mask == 0xFFFF) {
            p += 16;
        } else {
            p += __builtin_ctz((unsigned int)(~mask));
            break;
        }
    }
    *pp = p;
}

void skip_digits_avx2(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 32) {
            skip_digits_sse2(&p, end_pos);
            break;
        }
        __m256i data = _mm256_loadu_si256((const __m256i *)p);
        __m256i ge0 = _mm256_cmpgt_epi8(data, _mm256_set1_epi8('0' - 1));
        __m256i le9 = _mm256_cmpgt_epi8(_mm256_set1_epi8('9' + 1), data);
        __m256i is_dig = _mm256_and_si256(ge0, le9);
        uint32_t mask = (uint32_t)_mm256_movemask_epi8(is_dig);
        if (mask == 0xFFFFFFFFU) {
            p += 32;
        } else {
            p += __builtin_ctz((unsigned int)(~mask));
            break;
        }
    }
    *pp = p;
}

void skip_post_object_sse2(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 16) {
            while (p < end_pos && (is_space_lut[(unsigned char)*p] || *p == ',')) p++;
            break;
        }
        __m128i data = _mm_loadu_si128((const __m128i *)p);
        __m128i sp = _mm_cmpeq_epi8(data, _mm_set1_epi8(' '));
        __m128i tab = _mm_cmpeq_epi8(data, _mm_set1_epi8('\t'));
        __m128i nl = _mm_cmpeq_epi8(data, _mm_set1_epi8('\n'));
        __m128i cr = _mm_cmpeq_epi8(data, _mm_set1_epi8('\r'));
        __m128i vt = _mm_cmpeq_epi8(data, _mm_set1_epi8('\v'));
        __m128i ff = _mm_cmpeq_epi8(data, _mm_set1_epi8('\f'));
        __m128i comma = _mm_cmpeq_epi8(data, _mm_set1_epi8(','));
        __m128i is_skip = _mm_or_si128(sp, tab);
        is_skip = _mm_or_si128(is_skip, nl);
        is_skip = _mm_or_si128(is_skip, cr);
        is_skip = _mm_or_si128(is_skip, vt);
        is_skip = _mm_or_si128(is_skip, ff);
        is_skip = _mm_or_si128(is_skip, comma);
        int mask = _mm_movemask_epi8(is_skip);
        if (mask == 0xFFFF) {
            p += 16;
        } else {
            p += __builtin_ctz((unsigned int)(~mask));
            break;
        }
    }
    *pp = p;
}

void skip_post_object_avx2(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 32) {
            skip_post_object_sse2(&p, end_pos);
            break;
        }
        __m256i data = _mm256_loadu_si256((const __m256i *)p);
        __m256i sp = _mm256_cmpeq_epi8(data, _mm256_set1_epi8(' '));
        __m256i tab = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('\t'));
        __m256i nl = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('\n'));
        __m256i cr = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('\r'));
        __m256i vt = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('\v'));
        __m256i ff = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('\f'));
        __m256i comma = _mm256_cmpeq_epi8(data, _mm256_set1_epi8(','));
        __m256i is_skip = _mm256_or_si256(sp, tab);
        is_skip = _mm256_or_si256(is_skip, nl);
        is_skip = _mm256_or_si256(is_skip, cr);
        is_skip = _mm256_or_si256(is_skip, vt);
        is_skip = _mm256_or_si256(is_skip, ff);
        is_skip = _mm256_or_si256(is_skip, comma);
        uint32_t mask = (uint32_t)_mm256_movemask_epi8(is_skip);
        if (mask == 0xFFFFFFFFU) {
            p += 32;
        } else {
            p += __builtin_ctz((unsigned int)(~mask));
            break;
        }
    }
    *pp = p;
}

static __m128i get_is_safe_sse2(__m128i data) {
    __m128i az_min = _mm_set1_epi8('a' - 1);
    __m128i az_max = _mm_set1_epi8('z' + 1);
    __m128i AZ_min = _mm_set1_epi8('A' - 1);
    __m128i AZ_max = _mm_set1_epi8('Z' + 1);
    __m128i num_min = _mm_set1_epi8('0' - 1);
    __m128i num_max = _mm_set1_epi8('9' + 1);
    __m128i is_az = _mm_and_si128(_mm_cmpgt_epi8(data, az_min), _mm_cmplt_epi8(data, az_max));
    __m128i is_AZ = _mm_and_si128(_mm_cmpgt_epi8(data, AZ_min), _mm_cmplt_epi8(data, AZ_max));
    __m128i is_num = _mm_and_si128(_mm_cmpgt_epi8(data, num_min), _mm_cmplt_epi8(data, num_max));
    __m128i is_minus = _mm_cmpeq_epi8(data, _mm_set1_epi8('-'));
    __m128i is_dot = _mm_cmpeq_epi8(data, _mm_set1_epi8('.'));
    __m128i is_underscore = _mm_cmpeq_epi8(data, _mm_set1_epi8('_'));
    __m128i is_tilde = _mm_cmpeq_epi8(data, _mm_set1_epi8('~'));
    __m128i is_safe = _mm_or_si128(is_az, is_AZ);
    is_safe = _mm_or_si128(is_safe, is_num);
    is_safe = _mm_or_si128(is_safe, is_minus);
    is_safe = _mm_or_si128(is_safe, is_dot);
    is_safe = _mm_or_si128(is_safe, is_underscore);
    is_safe = _mm_or_si128(is_safe, is_tilde);
    return is_safe;
}

static __m256i get_is_safe_avx2(__m256i data) {
    __m256i az_min = _mm256_set1_epi8('a' - 1);
    __m256i az_max = _mm256_set1_epi8('z' + 1);
    __m256i AZ_min = _mm256_set1_epi8('A' - 1);
    __m256i AZ_max = _mm256_set1_epi8('Z' + 1);
    __m256i num_min = _mm256_set1_epi8('0' - 1);
    __m256i num_max = _mm256_set1_epi8('9' + 1);
    __m256i is_az = _mm256_and_si256(_mm256_cmpgt_epi8(data, az_min), _mm256_cmpgt_epi8(az_max, data));
    __m256i is_AZ = _mm256_and_si256(_mm256_cmpgt_epi8(data, AZ_min), _mm256_cmpgt_epi8(AZ_max, data));
    __m256i is_num = _mm256_and_si256(_mm256_cmpgt_epi8(data, num_min), _mm256_cmpgt_epi8(num_max, data));
    __m256i is_minus = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('-'));
    __m256i is_dot = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('.'));
    __m256i is_underscore = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('_'));
    __m256i is_tilde = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('~'));
    __m256i is_safe = _mm256_or_si256(is_az, is_AZ);
    is_safe = _mm256_or_si256(is_safe, is_num);
    is_safe = _mm256_or_si256(is_safe, is_minus);
    is_safe = _mm256_or_si256(is_safe, is_dot);
    is_safe = _mm256_or_si256(is_safe, is_underscore);
    is_safe = _mm256_or_si256(is_safe, is_tilde);
    return is_safe;
}

Str url_encode_sse2(const char *str, size_t len, struct Arena *__restrict a) {
    if (!str || len == 0) return (Str){NULL, 0};
    size_t buf_size = len * 3 + 1;
    char *buf = arena_alloc(a, buf_size);
    if (!buf) return (Str){NULL, 0};
    char *p = buf;
    const unsigned char *us = (const unsigned char *)str;
    const unsigned char *end = us + len;
    while (us < end) {
        ptrdiff_t rem = end - us;
        if (rem < 16) {
            while (us < end) {
                unsigned char c = *us++;
                if (is_url_safe[c]) {
                    *p++ = (char)c;
                } else {
                    *p++ = '%';
                    *p++ = hex_digits[c >> 4];
                    *p++ = hex_digits[c & 0xF];
                }
            }
            break;
        }
        __m128i data = _mm_loadu_si128((const __m128i *)us);
        __m128i is_safe = get_is_safe_sse2(data);
        int mask = _mm_movemask_epi8(is_safe);
        if (mask == 0xFFFF) {
            _mm_storeu_si128((__m128i *)p, data);
            p += 16;
            us += 16;
        } else {
            uint16_t not_safe = ~(uint16_t)mask;
            size_t pos = 0;
            while (not_safe) {
                unsigned int tz = (unsigned int)__builtin_ctz((unsigned int)not_safe);
                p = mempcpy_inline(p, (const char *)(us + pos), tz - pos);
                unsigned char c = us[tz];
                *p++ = '%';
                *p++ = hex_digits[c >> 4];
                *p++ = hex_digits[c & 0xF];
                pos = tz + 1;
                not_safe &= ~(1u << tz);
            }
            p = mempcpy_inline(p, (const char *)(us + pos), 16 - pos);
            us += 16;
        }
    }
    return (Str){buf, (size_t)(p - buf)};
}

Str url_encode_avx2(const char *str, size_t len, struct Arena *__restrict a) {
    if (!str || len == 0) return (Str){NULL, 0};
    size_t buf_size = len * 3 + 1;
    char *buf = arena_alloc(a, buf_size);
    if (!buf) return (Str){NULL, 0};
    char *p = buf;
    const unsigned char *us = (const unsigned char *)str;
    const unsigned char *end = us + len;
    while (us < end) {
        ptrdiff_t rem = end - us;
        if (rem < 32) {
            while (us < end) {
                unsigned char c = *us++;
                if (is_url_safe[c]) {
                    *p++ = (char)c;
                } else {
                    *p++ = '%';
                    *p++ = hex_digits[c >> 4];
                    *p++ = hex_digits[c & 0xF];
                }
            }
            break;
        }
        __m256i data = _mm256_loadu_si256((const __m256i *)us);
        __m256i is_safe = get_is_safe_avx2(data);
        uint32_t mask = (uint32_t)_mm256_movemask_epi8(is_safe);
        if (mask == 0xFFFFFFFFU) {
            _mm256_storeu_si256((__m256i *)p, data);
            p += 32;
            us += 32;
        } else {
            uint32_t not_safe = ~mask;
            size_t pos = 0;
            while (not_safe) {
                unsigned int tz = (unsigned int)__builtin_ctz(not_safe);
                p = mempcpy_inline(p, (const char *)(us + pos), tz - pos);
                unsigned char c = us[tz];
                *p++ = '%';
                *p++ = hex_digits[c >> 4];
                *p++ = hex_digits[c & 0xF];
                pos = tz + 1;
                not_safe &= ~(1u << tz);
            }
            p = mempcpy_inline(p, (const char *)(us + pos), 32 - pos);
            us += 32;
        }
    }
    return (Str){buf, (size_t)(p - buf)};
}

Str to_lower_sse2(const char *str, size_t len, struct Arena *__restrict a) {
    if (!str || len == 0) return (Str){NULL, 0};
    char *lower = arena_alloc(a, len + 1);
    if (!lower) return (Str){NULL, 0};
    size_t i = 0;
    for (; i + 16 <= len; i += 16) {
        __m128i chars = _mm_loadu_si128((const __m128i *)(str + i));
        __m128i a_min = _mm_set1_epi8('A' - 1);
        __m128i z_max = _mm_set1_epi8('Z' + 1);
        __m128i is_upper = _mm_and_si128(_mm_cmpgt_epi8(chars, a_min), _mm_cmplt_epi8(chars, z_max));
        __m128i add32 = _mm_and_si128(is_upper, _mm_set1_epi8(32));
        __m128i lowered = _mm_add_epi8(chars, add32);
        _mm_storeu_si128((__m128i *)(lower + i), lowered);
    }
    for (; i < len; i++) {
        char c = str[i];
        lower[i] = (c >= 'A' && c <= 'Z') ? c + 32 : c;
    }
    return (Str){lower, len};
}

Str to_lower_avx2(const char *str, size_t len, struct Arena *__restrict a) {
    if (!str || len == 0) return (Str){NULL, 0};
    char *lower = arena_alloc(a, len + 1);
    if (!lower) return (Str){NULL, 0};
    size_t i = 0;
    for (; i + 32 <= len; i += 32) {
        __m256i chars = _mm256_loadu_si256((const __m256i *)(str + i));
        __m256i a_min = _mm256_set1_epi8('A' - 1);
        __m256i z_max = _mm256_set1_epi8('Z' + 1);
        __m256i is_upper = _mm256_and_si256(_mm256_cmpgt_epi8(chars, a_min), _mm256_cmpgt_epi8(z_max, chars));
        __m256i add32 = _mm256_and_si256(is_upper, _mm256_set1_epi8(32));
        __m256i lowered = _mm256_add_epi8(chars, add32);
        _mm256_storeu_si256((__m256i *)(lower + i), lowered);
    }
    for (; i + 16 <= len; i += 16) {
        __m128i chars = _mm_loadu_si128((const __m128i *)(str + i));
        __m128i a_min = _mm_set1_epi8('A' - 1);
        __m128i z_max = _mm_set1_epi8('Z' + 1);
        __m128i is_upper = _mm_and_si128(_mm_cmpgt_epi8(chars, a_min), _mm_cmplt_epi8(chars, z_max));
        __m128i add32 = _mm_and_si128(is_upper, _mm_set1_epi8(32));
        __m128i lowered = _mm_add_epi8(chars, add32);
        _mm_storeu_si128((__m128i *)(lower + i), lowered);
    }
    for (; i < len; i++) {
        char c = str[i];
        lower[i] = (c >= 'A' && c <= 'Z') ? c + 32 : c;
    }
    return (Str){lower, len};
}
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
void skip_whitespace_avx512(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 64) {
            skip_whitespace_avx2(pp, end_pos);
            break;
        }
        __m512i data = _mm512_loadu_si512((const __m512i *)p);
        __mmask64 sp = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8(' '));
        __mmask64 tab = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8('\t'));
        __mmask64 nl = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8('\n'));
        __mmask64 cr = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8('\r'));
        __mmask64 vt = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8('\v'));
        __mmask64 ff = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8('\f'));
        __mmask64 is_ws = sp | tab | nl | cr | vt | ff;
        uint64_t mask = is_ws;
        if (mask == 0xFFFFFFFFFFFFFFFFULL) {
            p += 64;
        } else {
            p += __builtin_ctzll(~mask);
            break;
        }
    }
    *pp = p;
}

void find_quote_avx512(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 64) {
            find_quote_avx2(pp, end_pos);
            break;
        }
        __m512i data = _mm512_loadu_si512((const __m512i *)p);
        __mmask64 quotes = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8('"'));
        uint64_t mask = quotes;
        if (mask != 0) {
            p += __builtin_ctzll(mask);
            break;
        }
        p += 64;
    }
    *pp = p;
}

void skip_digits_avx512(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 64) {
            skip_digits_avx2(pp, end_pos);
            break;
        }
        __m512i data = _mm512_loadu_si512((const __m512i *)p);
        __mmask64 ge0 = _mm512_cmpgt_epi8_mask(data, _mm512_set1_epi8('0' - 1));
        __mmask64 le9 = _mm512_cmpgt_epi8_mask(_mm512_set1_epi8('9' + 1), data);
        __mmask64 is_dig = ge0 & le9;
        uint64_t mask = is_dig;
        if (mask == 0xFFFFFFFFFFFFFFFFULL) {
            p += 64;
        } else {
            p += __builtin_ctzll(~mask);
            break;
        }
    }
    *pp = p;
}

void skip_post_object_avx512(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 64) {
            skip_post_object_avx2(pp, end_pos);
            break;
        }
        __m512i data = _mm512_loadu_si512((const __m512i *)p);
        __mmask64 sp = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8(' '));
        __mmask64 tab = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8('\t'));
        __mmask64 nl = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8('\n'));
        __mmask64 cr = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8('\r'));
        __mmask64 vt = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8('\v'));
        __mmask64 ff = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8('\f'));
        __mmask64 comma = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8(','));
        __mmask64 is_ws = sp | tab | nl | cr | vt | ff;
        __mmask64 is_skip = is_ws | comma;
        uint64_t mask = is_skip;
        if (mask == 0xFFFFFFFFFFFFFFFFULL) {
            p += 64;
        } else {
            p += __builtin_ctzll(~mask);
            break;
        }
    }
    *pp = p;
}

static __mmask64 get_is_safe_avx512(__m512i data) {
    __m512i az_min = _mm512_set1_epi8('a' - 1);
    __m512i az_max = _mm512_set1_epi8('z' + 1);
    __m512i AZ_min = _mm512_set1_epi8('A' - 1);
    __m512i AZ_max = _mm512_set1_epi8('Z' + 1);
    __m512i num_min = _mm512_set1_epi8('0' - 1);
    __m512i num_max = _mm512_set1_epi8('9' + 1);
    __mmask64 is_az = _mm512_cmpgt_epi8_mask(data, az_min) & _mm512_cmpgt_epi8_mask(az_max, data);
    __mmask64 is_AZ = _mm512_cmpgt_epi8_mask(data, AZ_min) & _mm512_cmpgt_epi8_mask(AZ_max, data);
    __mmask64 is_num = _mm512_cmpgt_epi8_mask(data, num_min) & _mm512_cmpgt_epi8_mask(num_max, data);
    __mmask64 is_minus = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8('-'));
    __mmask64 is_dot = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8('.'));
    __mmask64 is_underscore = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8('_'));
    __mmask64 is_tilde = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8('~'));
    __mmask64 is_safe = is_az | is_AZ;
    is_safe = is_safe | is_num;
    is_safe = is_safe | is_minus;
    is_safe = is_safe | is_dot;
    is_safe = is_safe | is_underscore;
    is_safe = is_safe | is_tilde;
    return is_safe;
}

Str url_encode_avx512(const char *str, size_t len, struct Arena *__restrict a) {
    if (!str || len == 0) return (Str){NULL, 0};
    size_t buf_size = len * 3 + 1;
    char *buf = arena_alloc(a, buf_size);
    if (!buf) return (Str){NULL, 0};
    char *p = buf;
    const unsigned char *us = (const unsigned char *)str;
    const unsigned char *end = us + len;
    while (us < end) {
        ptrdiff_t rem = end - us;
        if (rem < 64) {
            while (us < end) {
                unsigned char c = *us++;
                if (is_url_safe[c]) {
                    *p++ = (char)c;
                } else {
                    *p++ = '%';
                    *p++ = hex_digits[c >> 4];
                    *p++ = hex_digits[c & 0xF];
                }
            }
            break;
        }
        __m512i data = _mm512_loadu_si512((const __m512i *)us);
        __mmask64 is_safe = get_is_safe_avx512(data);
        uint64_t mask = is_safe;
        if (mask == 0xFFFFFFFFFFFFFFFFULL) {
            _mm512_storeu_si512((__m512i *)p, data);
            p += 64;
            us += 64;
        } else {
            uint64_t not_safe = ~mask;
            size_t pos = 0;
            while (not_safe) {
                unsigned int tz = (unsigned int)__builtin_ctzll(not_safe);
                p = mempcpy_inline(p, (const char *)(us + pos), tz - pos);
                unsigned char c = us[tz];
                *p++ = '%';
                *p++ = hex_digits[c >> 4];
                *p++ = hex_digits[c & 0xF];
                pos = tz + 1;
                not_safe &= ~(1ULL << tz);
            }
            p = mempcpy_inline(p, (const char *)(us + pos), 64 - pos);
            us += 64;
        }
    }
    return (Str){buf, (size_t)(p - buf)};
}

Str to_lower_avx512(const char *str, size_t len, struct Arena *__restrict a) {
    if (!str || len == 0) return (Str){NULL, 0};
    char *lower = arena_alloc(a, len + 1);
    if (!lower) return (Str){NULL, 0};
    size_t i = 0;
    for (; i + 64 <= len; i += 64) {
        __m512i chars = _mm512_loadu_si512((const __m512i *)(str + i));
        __m512i a_min = _mm512_set1_epi8('A' - 1);
        __m512i z_max = _mm512_set1_epi8('Z' + 1);
        __mmask64 is_upper = _mm512_cmpgt_epi8_mask(chars, a_min) & _mm512_cmpgt_epi8_mask(z_max, chars);
        __m512i add32 = _mm512_mask_blend_epi8(is_upper, _mm512_setzero_si512(), _mm512_set1_epi8(32));
        __m512i lowered = _mm512_add_epi8(chars, add32);
        _mm512_storeu_si512((__m512i *)(lower + i), lowered);
    }
    for (; i + 32 <= len; i += 32) {
        __m256i chars = _mm256_loadu_si256((const __m256i *)(str + i));
        __m256i a_min = _mm256_set1_epi8('A' - 1);
        __m256i z_max = _mm256_set1_epi8('Z' + 1);
        __m256i is_upper = _mm256_and_si256(_mm256_cmpgt_epi8(chars, a_min), _mm256_cmpgt_epi8(z_max, chars));
        __m256i add32 = _mm256_and_si256(is_upper, _mm256_set1_epi8(32));
        __m256i lowered = _mm256_add_epi8(chars, add32);
        _mm256_storeu_si256((__m256i *)(lower + i), lowered);
    }
    for (; i + 16 <= len; i += 16) {
        __m128i chars = _mm_loadu_si128((const __m128i *)(str + i));
        __m128i a_min = _mm_set1_epi8('A' - 1);
        __m128i z_max = _mm_set1_epi8('Z' + 1);
        __m128i is_upper = _mm_and_si128(_mm_cmpgt_epi8(chars, a_min), _mm_cmplt_epi8(chars, z_max));
        __m128i add32 = _mm_and_si128(is_upper, _mm_set1_epi8(32));
        __m128i lowered = _mm_add_epi8(chars, add32);
        _mm_storeu_si128((__m128i *)(lower + i), lowered);
    }
    for (; i < len; i++) {
        char c = str[i];
        lower[i] = (c >= 'A' && c <= 'Z') ? c + 32 : c;
    }
    return (Str){lower, len};
}
#endif

#ifdef HAS_ARM
static uint8x16_t get_is_safe_neon(uint8x16_t data) {
    uint8x16_t az_min = vdupq_n_u8('a' - 1);
    uint8x16_t az_max = vdupq_n_u8('z' + 1);
    uint8x16_t AZ_min = vdupq_n_u8('A' - 1);
    uint8x16_t AZ_max = vdupq_n_u8('Z' + 1);
    uint8x16_t num_min = vdupq_n_u8('0' - 1);
    uint8x16_t num_max = vdupq_n_u8('9' + 1);
    uint8x16_t is_az = vandq_u8(vcgtq_u8(data, az_min), vcltq_u8(data, az_max));
    uint8x16_t is_AZ = vandq_u8(vcgtq_u8(data, AZ_min), vcltq_u8(data, AZ_max));
    uint8x16_t is_num = vandq_u8(vcgtq_u8(data, num_min), vcltq_u8(data, num_max));
    uint8x16_t is_minus = vceqq_u8(data, vdupq_n_u8('-'));
    uint8x16_t is_dot = vceqq_u8(data, vdupq_n_u8('.'));
    uint8x16_t is_underscore = vceqq_u8(data, vdupq_n_u8('_'));
    uint8x16_t is_tilde = vceqq_u8(data, vdupq_n_u8('~'));
    uint8x16_t is_safe = vorrq_u8(is_az, is_AZ);
    is_safe = vorrq_u8(is_safe, is_num);
    is_safe = vorrq_u8(is_safe, is_minus);
    is_safe = vorrq_u8(is_safe, is_dot);
    is_safe = vorrq_u8(is_safe, is_underscore);
    is_safe = vorrq_u8(is_safe, is_tilde);
    return is_safe;
}

void skip_whitespace_neon(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 16) {
            skip_whitespace_scalar(pp, end_pos);
            break;
        }
        uint8x16_t data = vld1q_u8((uint8_t *)p);
        uint8x16_t sp = vceqq_u8(data, vdupq_n_u8(' '));
        uint8x16_t tab = vceqq_u8(data, vdupq_n_u8('\t'));
        uint8x16_t nl = vceqq_u8(data, vdupq_n_u8('\n'));
        uint8x16_t cr = vceqq_u8(data, vdupq_n_u8('\r'));
        uint8x16_t vt = vceqq_u8(data, vdupq_n_u8('\v'));
        uint8x16_t ff = vceqq_u8(data, vdupq_n_u8('\f'));
        uint8x16_t is_ws = vorrq_u8(vorrq_u8(vorrq_u8(sp, tab), vorrq_u8(nl, cr)), vorrq_u8(vt, ff));
        uint8x16_t mvn = vmvnq_u8(is_ws);
        uint8x16_t bits = vshrq_n_u8(mvn, 7);
        uint64x2_t mask64 = vreinterpretq_u64_u8(bits);
        uint64_t low = vgetq_lane_u64(mask64, 0);
        uint64_t high = vgetq_lane_u64(mask64, 1);
        if (low == 0 && high == 0) {
            p += 16;
        } else {
            int pos;
            if (low != 0) {
                pos = __builtin_ctzll(low) / 8;
            } else {
                pos = 8 + __builtin_ctzll(high) / 8;
            }
            p += pos;
            break;
        }
    }
    *pp = p;
}

void find_quote_neon(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 16) {
            find_quote_scalar(pp, end_pos);
            break;
        }
        uint8x16_t data = vld1q_u8((uint8_t *)p);
        uint8x16_t quotes = vceqq_u8(data, vdupq_n_u8('"'));
        uint8x16_t bits = vshrq_n_u8(quotes, 7);
        uint64x2_t mask64 = vreinterpretq_u64_u8(bits);
        uint64_t low = vgetq_lane_u64(mask64, 0);
        uint64_t high = vgetq_lane_u64(mask64, 1);
        if (low == 0 && high == 0) {
            p += 16;
        } else {
            int pos;
            if (low != 0) {
                pos = __builtin_ctzll(low) / 8;
            } else {
                pos = 8 + __builtin_ctzll(high) / 8;
            }
            p += pos;
            break;
        }
    }
    *pp = p;
}

void skip_digits_neon(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 16) {
            skip_digits_scalar(pp, end_pos);
            break;
        }
        uint8x16_t data = vld1q_u8((uint8_t *)p);
        uint8x16_t ge0 = vcgtq_u8(data, vdupq_n_u8('0' - 1));
        uint8x16_t le9 = vcltq_u8(data, vdupq_n_u8('9' + 1));
        uint8x16_t is_dig = vandq_u8(ge0, le9);
        uint8x16_t mvn = vmvnq_u8(is_dig);
        uint8x16_t bits = vshrq_n_u8(mvn, 7);
        uint64x2_t mask64 = vreinterpretq_u64_u8(bits);
        uint64_t low = vgetq_lane_u64(mask64, 0);
        uint64_t high = vgetq_lane_u64(mask64, 1);
        if (low == 0 && high == 0) {
            p += 16;
        } else {
            int pos;
            if (low != 0) {
                pos = __builtin_ctzll(low) / 8;
            } else {
                pos = 8 + __builtin_ctzll(high) / 8;
            }
            p += pos;
            break;
        }
    }
    *pp = p;
}

void skip_post_object_neon(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 16) {
            skip_post_object_scalar(pp, end_pos);
            break;
        }
        uint8x16_t data = vld1q_u8((uint8_t *)p);
        uint8x16_t sp = vceqq_u8(data, vdupq_n_u8(' '));
        uint8x16_t tab = vceqq_u8(data, vdupq_n_u8('\t'));
        uint8x16_t nl = vceqq_u8(data, vdupq_n_u8('\n'));
        uint8x16_t cr = vceqq_u8(data, vdupq_n_u8('\r'));
        uint8x16_t vt = vceqq_u8(data, vdupq_n_u8('\v'));
        uint8x16_t ff = vceqq_u8(data, vdupq_n_u8('\f'));
        uint8x16_t comma = vceqq_u8(data, vdupq_n_u8(','));
        uint8x16_t is_skip = vorrq_u8(vorrq_u8(vorrq_u8(sp, tab), vorrq_u8(nl, cr)), vorrq_u8(vorrq_u8(vt, ff), comma));
        uint8x16_t mvn = vmvnq_u8(is_skip);
        uint8x16_t bits = vshrq_n_u8(mvn, 7);
        uint64x2_t mask64 = vreinterpretq_u64_u8(bits);
        uint64_t low = vgetq_lane_u64(mask64, 0);
        uint64_t high = vgetq_lane_u64(mask64, 1);
        if (low == 0 && high == 0) {
            p += 16;
        } else {
            int pos;
            if (low != 0) {
                pos = __builtin_ctzll(low) / 8;
            } else {
                pos = 8 + __builtin_ctzll(high) / 8;
            }
            p += pos;
            break;
        }
    }
    *pp = p;
}

Str url_encode_neon(const char *str, size_t len, struct Arena *__restrict a) {
    if (!str || len == 0) return (Str){NULL, 0};
    size_t buf_size = len * 3 + 1;
    char *buf = arena_alloc(a, buf_size);
    if (!buf) return (Str){NULL, 0};
    char *p = buf;
    const unsigned char *us = (const unsigned char *)str;
    const unsigned char *end = us + len;
    while (us < end) {
        ptrdiff_t rem = end - us;
        if (rem < 16) {
            while (us < end) {
                unsigned char c = *us++;
                if (is_url_safe[c]) {
                    *p++ = (char)c;
                } else {
                    *p++ = '%';
                    *p++ = hex_digits[c >> 4];
                    *p++ = hex_digits[c & 0xF];
                }
            }
            break;
        }
        uint8x16_t data = vld1q_u8(us);
        uint8x16_t is_safe = get_is_safe_neon(data);
        uint8x16_t mvn = vmvnq_u8(is_safe);
        uint8x16_t sign_bits = vshrq_n_u8(mvn, 7);
        uint64x2_t mask64 = vreinterpretq_u64_u8(sign_bits);
        uint64_t low = vgetq_lane_u64(mask64, 0);
        uint64_t high = vgetq_lane_u64(mask64, 1);
        if (low == 0 && high == 0) {
            vst1q_u8((uint8_t *)p, data);
            p += 16;
            us += 16;
        } else {
            uint16_t not_safe = 0;
            for (int i = 0; i < 16; i++) {
                if (vgetq_lane_u8(sign_bits, i)) not_safe |= (1u << i);
            }
            size_t pos = 0;
            while (not_safe) {
                unsigned int tz = __builtin_ctz(not_safe);
                p = mempcpy_inline(p, (const char *)(us + pos), tz - pos);
                unsigned char c = us[tz];
                *p++ = '%';
                *p++ = hex_digits[c >> 4];
                *p++ = hex_digits[c & 0xF];
                pos = tz + 1;
                not_safe &= ~(1u << tz);
            }
            p = mempcpy_inline(p, (const char *)(us + pos), 16 - pos);
            us += 16;
        }
    }
    return (Str){buf, (size_t)(p - buf)};
}

Str to_lower_neon(const char *str, size_t len, struct Arena *__restrict a) {
    if (!str || len == 0) return (Str){NULL, 0};
    char *lower = arena_alloc(a, len + 1);
    if (!lower) return (Str){NULL, 0};
    size_t i = 0;
    for (; i + 16 <= len; i += 16) {
        uint8x16_t chars = vld1q_u8((uint8_t *)(str + i));
        uint8x16_t a_min = vdupq_n_u8('A' - 1);
        uint8x16_t z_max = vdupq_n_u8('Z' + 1);
        uint8x16_t is_upper = vandq_u8(vcgtq_u8(chars, a_min), vcltq_u8(chars, z_max));
        uint8x16_t add32 = vandq_u8(is_upper, vdupq_n_u8(32));
        uint8x16_t lowered = vaddq_u8(chars, add32);
        vst1q_u8((uint8_t *)(lower + i), lowered);
    }
    for (; i < len; i++) {
        char c = str[i];
        lower[i] = (c >= 'A' && c <= 'Z') ? c + 32 : c;
    }
    return (Str){lower, len};
}
#endif

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
void skip_whitespace_sve(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        svbool_t pg = svwhilelt_b8(0, remaining);
        if (svptest_any(svptrue_b8(), pg) == 0) break;
        svuint8_t data = svld1(pg, (const uint8_t *)p);
        svbool_t sp = svcmpeq(pg, data, ' ');
        svbool_t tab = svcmpeq(pg, data, '\t');
        svbool_t nl = svcmpeq(pg, data, '\n');
        svbool_t cr = svcmpeq(pg, data, '\r');
        svbool_t vt = svcmpeq(pg, data, '\v');
        svbool_t ff = svcmpeq(pg, data, '\f');
        svbool_t is_ws = svorr_b_z(pg, svorr_b_z(pg, svorr_b_z(pg, sp, tab), svorr_b_z(pg, nl, cr)), svorr_b_z(pg, vt, ff));
        svbool_t non_ws = svnot_b_z(pg, is_ws);
        if (svptest_any(svptrue_b8(), non_ws) == 0) {
            p += svcntb();
        } else {
            int first = svcntp_b8(pg, svbrka_b_z(pg, non_ws));
            p += first - 1;
            break;
        }
    }
    *pp = p;
}

void find_quote_sve(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        svbool_t pg = svwhilelt_b8(0, remaining);
        if (svptest_any(svptrue_b8(), pg) == 0) break;
        svuint8_t data = svld1(pg, (const uint8_t *)p);
        svbool_t quotes = svcmpeq(pg, data, '"');
        if (svptest_any(svptrue_b8(), quotes) == 0) {
            p += svcntb();
        } else {
            int first = svcntp_b8(pg, svbrka_b_z(pg, quotes));
            p += first - 1;
            break;
        }
    }
    *pp = p;
}

void skip_digits_sve(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        svbool_t pg = svwhilelt_b8(0, remaining);
        if (svptest_any(svptrue_b8(), pg) == 0) break;
        svuint8_t data = svld1(pg, (const uint8_t *)p);
        svbool_t ge0 = svcmpgt(pg, data, '0' - 1);
        svbool_t le9 = svcmpgt(pg, '9' + 1, data);
        svbool_t is_dig = svand_b_z(pg, ge0, le9);
        svbool_t non_dig = svnot_b_z(pg, is_dig);
        if (svptest_any(svptrue_b8(), non_dig) == 0) {
            p += svcntb();
        } else {
            int first = svcntp_b8(pg, svbrka_b_z(pg, non_dig));
            p += first - 1;
            break;
        }
    }
    *pp = p;
}

void skip_post_object_sve(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        svbool_t pg = svwhilelt_b8(0, remaining);
        if (svptest_any(svptrue_b8(), pg) == 0) break;
        svuint8_t data = svld1(pg, (const uint8_t *)p);
        svbool_t sp = svcmpeq(pg, data, ' ');
        svbool_t tab = svcmpeq(pg, data, '\t');
        svbool_t nl = svcmpeq(pg, data, '\n');
        svbool_t cr = svcmpeq(pg, data, '\r');
        svbool_t vt = svcmpeq(pg, data, '\v');
        svbool_t ff = svcmpeq(pg, data, '\f');
        svbool_t comma = svcmpeq(pg, data, ',');
        svbool_t is_ws = svorr_b_z(pg, svorr_b_z(pg, svorr_b_z(pg, sp, tab), svorr_b_z(pg, nl, cr)), svorr_b_z(pg, vt, ff));
        svbool_t is_skip = svorr_b_z(pg, is_ws, comma);
        svbool_t non_skip = svnot_b_z(pg, is_skip);
        if (svptest_any(svptrue_b8(), non_skip) == 0) {
            p += svcntb();
        } else {
            int first = svcntp_b8(pg, svbrka_b_z(pg, non_skip));
            p += first - 1;
            break;
        }
    }
    *pp = p;
}

static svbool_t get_is_safe_sve(svuint8_t data, svbool_t pg) {
    svbool_t is_az = svand_b_z(pg, svcmpgt(pg, data, 'a' - 1), svcmpgt(pg, 'z' + 1, data));
    svbool_t is_AZ = svand_b_z(pg, svcmpgt(pg, data, 'A' - 1), svcmpgt(pg, 'Z' + 1, data));
    svbool_t is_num = svand_b_z(pg, svcmpgt(pg, data, '0' - 1), svcmpgt(pg, '9' + 1, data));
    svbool_t is_minus = svcmpeq(pg, data, '-');
    svbool_t is_dot = svcmpeq(pg, data, '.');
    svbool_t is_underscore = svcmpeq(pg, data, '_');
    svbool_t is_tilde = svcmpeq(pg, data, '~');
    svbool_t is_safe = svorr_b_z(pg, is_az, is_AZ);
    is_safe = svorr_b_z(pg, is_safe, is_num);
    is_safe = svorr_b_z(pg, is_safe, is_minus);
    is_safe = svorr_b_z(pg, is_safe, is_dot);
    is_safe = svorr_b_z(pg, is_safe, is_underscore);
    is_safe = svorr_b_z(pg, is_safe, is_tilde);
    return is_safe;
}

Str url_encode_sve(const char *str, size_t len, struct Arena *__restrict a) {
    if (!str || len == 0) return (Str){NULL, 0};
    size_t buf_size = len * 3 + 1;
    char *buf = arena_alloc(a, buf_size);
    if (!buf) return (Str){NULL, 0};
    char *p = buf;
    const unsigned char *us = (const unsigned char *)str;
    const unsigned char *end = us + len;
    while (us < end) {
        ptrdiff_t rem = end - us;
        svbool_t pg = svwhilelt_b8(0, rem);
        if (svptest_any(svptrue_b8(), pg) == 0) break;
        svuint8_t data = svld1(pg, us);
        svbool_t is_safe = get_is_safe_sve(data, pg);
        svbool_t non_safe = svnot_b_z(pg, is_safe);
        if (svptest_any(svptrue_b8(), non_safe) == 0) {
            svst1(pg, (uint8_t *)p, data);
            p += svcntb();
            us += svcntb();
        } else {
            size_t vl = svcntb();
            for (size_t k = 0; k < vl; k++) {
                unsigned char c = us[k];
                if (is_url_safe[c]) {
                    *p++ = (char)c;
                } else {
                    *p++ = '%';
                    *p++ = hex_digits[c >> 4];
                    *p++ = hex_digits[c & 0xF];
                }
            }
            us += vl;
        }
    }
    return (Str){buf, (size_t)(p - buf)};
}

Str to_lower_sve(const char *str, size_t len, struct Arena *__restrict a) {
    if (!str || len == 0) return (Str){NULL, 0};
    char *lower = arena_alloc(a, len + 1);
    if (!lower) return (Str){NULL, 0};
    size_t i = 0;
    while (i < len) {
        svbool_t pg = svwhilelt_b8(i, len);
        svuint8_t chars = svld1(pg, (uint8_t *)(str + i));
        svbool_t is_upper = svand_b_z(pg, svcmpgt(pg, chars, 'A' - 1), svcmpgt(pg, 'Z' + 1, chars));
        svuint8_t add32 = svsel_u8(is_upper, svdup_u8(32), svdup_u8(0));
        svuint8_t lowered = svadd_u8_m(pg, chars, add32);
        svst1(pg, (uint8_t *)(lower + i), lowered);
        i += svcntb();
    }
    return (Str){lower, len};
}
#endif

#ifdef HAS_POWER
typedef vector unsigned char vec_u8_t;

static vec_u8_t get_is_safe_vsx(vec_u8_t data) {
    vec_u8_t az_min = vec_splats((unsigned char)('a' - 1));
    vec_u8_t az_max = vec_splats((unsigned char)('z' + 1));
    vec_u8_t AZ_min = vec_splats((unsigned char)('A' - 1));
    vec_u8_t AZ_max = vec_splats((unsigned char)('Z' + 1));
    vec_u8_t num_min = vec_splats((unsigned char)('0' - 1));
    vec_u8_t num_max = vec_splats((unsigned char)('9' + 1));
    vec_u8_t is_az = vec_and(vec_cmpgt(data, az_min), vec_cmpgt(az_max, data));
    vec_u8_t is_AZ = vec_and(vec_cmpgt(data, AZ_min), vec_cmpgt(AZ_max, data));
    vec_u8_t is_num = vec_and(vec_cmpgt(data, num_min), vec_cmpgt(num_max, data));
    vec_u8_t is_minus = vec_cmpeq(data, vec_splats((unsigned char)'-'));
    vec_u8_t is_dot = vec_cmpeq(data, vec_splats((unsigned char)'.'));
    vec_u8_t is_underscore = vec_cmpeq(data, vec_splats((unsigned char)'_'));
    vec_u8_t is_tilde = vec_cmpeq(data, vec_splats((unsigned char)'~'));
    vec_u8_t is_safe = vec_or(is_az, is_AZ);
    is_safe = vec_or(is_safe, is_num);
    is_safe = vec_or(is_safe, is_minus);
    is_safe = vec_or(is_safe, is_dot);
    is_safe = vec_or(is_safe, is_underscore);
    is_safe = vec_or(is_safe, is_tilde);
    return is_safe;
}

void skip_whitespace_vsx(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 16) {
            skip_whitespace_scalar(pp, end_pos);
            break;
        }
        vec_u8_t data = vec_ld(0, (unsigned char *)p);
        vec_u8_t sp = vec_cmpeq(data, vec_splats((unsigned char)' '));
        vec_u8_t tab = vec_cmpeq(data, vec_splats((unsigned char)'\t'));
        vec_u8_t nl = vec_cmpeq(data, vec_splats((unsigned char)'\n'));
        vec_u8_t cr = vec_cmpeq(data, vec_splats((unsigned char)'\r'));
        vec_u8_t vt = vec_cmpeq(data, vec_splats((unsigned char)'\v'));
        vec_u8_t ff = vec_cmpeq(data, vec_splats((unsigned char)'\f'));
        vec_u8_t is_ws = vec_or(vec_or(vec_or(sp, tab), vec_or(nl, cr)), vec_or(vt, ff));
        if (vec_all_eq(is_ws, vec_splats((unsigned char)0xFF))) {
            p += 16;
        } else {
            for (int i = 0; i < 16; i++) {
                if (!is_space_lut[(unsigned char)p[i]]) {
                    p += i;
                    break;
                }
            }
            break;
        }
    }
    *pp = p;
}

void find_quote_vsx(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 16) {
            find_quote_scalar(pp, end_pos);
            break;
        }
        vec_u8_t data = vec_ld(0, (unsigned char *)p);
        vec_u8_t quotes = vec_cmpeq(data, vec_splats((unsigned char)'"'));
        if (vec_any_eq(quotes, vec_splats((unsigned char)0xFF))) {
            for (int i = 0; i < 16; i++) {
                if (p[i] == '"') {
                    p += i;
                    break;
                }
            }
            break;
        } else {
            p += 16;
        }
    }
    *pp = p;
}

void skip_digits_vsx(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 16) {
            skip_digits_scalar(pp, end_pos);
            break;
        }
        vec_u8_t data = vec_ld(0, (unsigned char *)p);
        vec_u8_t ge0 = vec_cmpgt(data, vec_splats((unsigned char)('0' - 1)));
        vec_u8_t le9 = vec_cmpgt(vec_splats((unsigned char)('9' + 1)), data);
        vec_u8_t is_dig = vec_and(ge0, le9);
        if (vec_all_eq(is_dig, vec_splats((unsigned char)0xFF))) {
            p += 16;
        } else {
            for (int i = 0; i < 16; i++) {
                if (!is_digit_lut[(unsigned char)p[i]]) {
                    p += i;
                    break;
                }
            }
            break;
        }
    }
    *pp = p;
}

void skip_post_object_vsx(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 16) {
            skip_post_object_scalar(pp, end_pos);
            break;
        }
        vec_u8_t data = vec_ld(0, (unsigned char *)p);
        vec_u8_t sp = vec_cmpeq(data, vec_splats((unsigned char)' '));
        vec_u8_t tab = vec_cmpeq(data, vec_splats((unsigned char)'\t'));
        vec_u8_t nl = vec_cmpeq(data, vec_splats((unsigned char)'\n'));
        vec_u8_t cr = vec_cmpeq(data, vec_splats((unsigned char)'\r'));
        vec_u8_t vt = vec_cmpeq(data, vec_splats((unsigned char)'\v'));
        vec_u8_t ff = vec_cmpeq(data, vec_splats((unsigned char)'\f'));
        vec_u8_t comma = vec_cmpeq(data, vec_splats((unsigned char)','));
        vec_u8_t is_skip = vec_or(vec_or(vec_or(sp, tab), vec_or(nl, cr)), vec_or(vec_or(vt, ff), comma));
        if (vec_all_eq(is_skip, vec_splats((unsigned char)0xFF))) {
            p += 16;
        } else {
            for (int i = 0; i < 16; i++) {
                if (!(is_space_lut[(unsigned char)p[i]] || p[i] == ',')) {
                    p += i;
                    break;
                }
            }
            break;
        }
    }
    *pp = p;
}

Str url_encode_vsx(const char *str, size_t len, struct Arena *__restrict a) {
    if (!str || len == 0) return (Str){NULL, 0};
    size_t buf_size = len * 3 + 1;
    char *buf = arena_alloc(a, buf_size);
    if (!buf) return (Str){NULL, 0};
    char *p = buf;
    const unsigned char *us = (const unsigned char *)str;
    const unsigned char *end = us + len;
    while (us < end) {
        ptrdiff_t rem = end - us;
        if (rem < 16) {
            while (us < end) {
                unsigned char c = *us++;
                if (is_url_safe[c]) {
                    *p++ = (char)c;
                } else {
                    *p++ = '%';
                    *p++ = hex_digits[c >> 4];
                    *p++ = hex_digits[c & 0xF];
                }
            }
            break;
        }
        vec_u8_t data = vec_ld(0, (unsigned char *)us);
        vec_u8_t is_safe = get_is_safe_vsx(data);
        uint16_t not_safe = 0;
        for (int i = 0; i < 16; i++) {
            if (is_safe[i] == 0) not_safe |= (1u << i);
        }
        if (not_safe == 0) {
            vec_st(data, 0, (unsigned char *)p);
            p += 16;
            us += 16;
        } else {
            size_t pos = 0;
            while (not_safe) {
                unsigned int tz = __builtin_ctz(not_safe);
                p = mempcpy_inline(p, (const char *)(us + pos), tz - pos);
                unsigned char c = us[tz];
                *p++ = '%';
                *p++ = hex_digits[c >> 4];
                *p++ = hex_digits[c & 0xF];
                pos = tz + 1;
                not_safe &= ~(1u << tz);
            }
            p = mempcpy_inline(p, (const char *)(us + pos), 16 - pos);
            us += 16;
        }
    }
    return (Str){buf, (size_t)(p - buf)};
}

Str to_lower_vsx(const char *str, size_t len, struct Arena *__restrict a) {
    if (!str || len == 0) return (Str){NULL, 0};
    char *lower = arena_alloc(a, len + 1);
    if (!lower) return (Str){NULL, 0};
    size_t i = 0;
    for (; i + 16 <= len; i += 16) {
        vec_u8_t chars = vec_ld(0, (unsigned char *)(str + i));
        vec_u8_t a_min = vec_splats((unsigned char)('A' - 1));
        vec_u8_t z_max = vec_splats((unsigned char)('Z' + 1));
        vec_u8_t is_upper = vec_and(vec_cmpgt(chars, a_min), vec_cmpgt(z_max, chars));
        vec_u8_t add32 = vec_sel(vec_splats((unsigned char)0), vec_splats((unsigned char)32), is_upper);
        vec_u8_t lowered = vec_add(chars, add32);
        vec_st(lowered, 0, (unsigned char *)(lower + i));
    }
    for (; i < len; i++) {
        char c = str[i];
        lower[i] = (c >= 'A' && c <= 'Z') ? c + 32 : c;
    }
    return (Str){lower, len};
}
#endif

#ifdef HAS_RISCV
void skip_whitespace_rvv(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 32) {
            skip_whitespace_scalar(pp, end_pos);
            break;
        }
        size_t vl = __riscv_vsetvl_e8m1(32);
        vuint8m1_t data = __riscv_vle8_v_u8m1((uint8_t *)p, vl);
        vbool8_t sp = __riscv_vmseq_vx_u8m1_b8(data, ' ', vl);
        vbool8_t tab = __riscv_vmseq_vx_u8m1_b8(data, '\t', vl);
        vbool8_t nl = __riscv_vmseq_vx_u8m1_b8(data, '\n', vl);
        vbool8_t cr = __riscv_vmseq_vx_u8m1_b8(data, '\r', vl);
        vbool8_t vt = __riscv_vmseq_vx_u8m1_b8(data, '\v', vl);
        vbool8_t ff = __riscv_vmseq_vx_u8m1_b8(data, '\f', vl);
        vbool8_t is_ws = __riscv_vmor_mm_b8(__riscv_vmor_mm_b8(__riscv_vmor_mm_b8(sp, tab, vl), __riscv_vmor_mm_b8(nl, cr, vl), vl), __riscv_vmor_mm_b8(vt, ff, vl), vl);
        vbool8_t is_non_ws = __riscv_vmnot_m_b8(is_ws, vl);
        int first = __riscv_vfirst_m_b8(is_non_ws, vl);
        if (first == -1) {
            p += vl;
        } else {
            p += first;
            break;
        }
    }
    *pp = p;
}

void find_quote_rvv(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 32) {
            find_quote_scalar(pp, end_pos);
            break;
        }
        size_t vl = __riscv_vsetvl_e8m1(32);
        vuint8m1_t data = __riscv_vle8_v_u8m1((uint8_t *)p, vl);
        vbool8_t quotes = __riscv_vmseq_vx_u8m1_b8(data, '"', vl);
        int first = __riscv_vfirst_m_b8(quotes, vl);
        if (first == -1) {
            p += vl;
        } else {
            p += first;
            break;
        }
    }
    *pp = p;
}

void skip_digits_rvv(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 32) {
            skip_digits_scalar(pp, end_pos);
            break;
        }
        size_t vl = __riscv_vsetvl_e8m1(32);
        vuint8m1_t data = __riscv_vle8_v_u8m1((uint8_t *)p, vl);
        vbool8_t ge0 = __riscv_vmsgtu_vx_u8m1_b8(data, '0' - 1, vl);
        vbool8_t le9 = __riscv_vmsleu_vx_u8m1_b8(data, '9', vl);
        vbool8_t is_dig = __riscv_vmand_mm_b8(ge0, le9, vl);
        vbool8_t non_dig = __riscv_vmnot_m_b8(is_dig, vl);
        int first = __riscv_vfirst_m_b8(non_dig, vl);
        if (first == -1) {
            p += vl;
        } else {
            p += first;
            break;
        }
    }
    *pp = p;
}

void skip_post_object_rvv(char **pp, const char *end_pos) {
    char *p = *pp;
    while (p < end_pos) {
        ptrdiff_t remaining = end_pos - p;
        if (remaining < 32) {
            skip_post_object_scalar(pp, end_pos);
            break;
        }
        size_t vl = __riscv_vsetvl_e8m1(32);
        vuint8m1_t data = __riscv_vle8_v_u8m1((uint8_t *)p, vl);
        vbool8_t sp = __riscv_vmseq_vx_u8m1_b8(data, ' ', vl);
        vbool8_t tab = __riscv_vmseq_vx_u8m1_b8(data, '\t', vl);
        vbool8_t nl = __riscv_vmseq_vx_u8m1_b8(data, '\n', vl);
        vbool8_t cr = __riscv_vmseq_vx_u8m1_b8(data, '\r', vl);
        vbool8_t vt = __riscv_vmseq_vx_u8m1_b8(data, '\v', vl);
        vbool8_t ff = __riscv_vmseq_vx_u8m1_b8(data, '\f', vl);
        vbool8_t comma = __riscv_vmseq_vx_u8m1_b8(data, ',', vl);
        vbool8_t is_ws = __riscv_vmor_mm_b8(__riscv_vmor_mm_b8(__riscv_vmor_mm_b8(sp, tab, vl), __riscv_vmor_mm_b8(nl, cr, vl), vl), __riscv_vmor_mm_b8(vt, ff, vl), vl);
        vbool8_t is_skip = __riscv_vmor_mm_b8(is_ws, comma, vl);
        vbool8_t non_skip = __riscv_vmnot_m_b8(is_skip, vl);
        int first = __riscv_vfirst_m_b8(non_skip, vl);
        if (first == -1) {
            p += vl;
        } else {
            p += first;
            break;
        }
    }
    *pp = p;
}

static vbool8_t get_is_safe_rvv(vuint8m1_t data, size_t vl) {
    vbool8_t is_az = __riscv_vmand_mm_b8(__riscv_vmsgtu_vx_u8m1_b8(data, 'a' - 1, vl), __riscv_vmsleu_vx_u8m1_b8(data, 'z', vl), vl);
    vbool8_t is_AZ = __riscv_vmand_mm_b8(__riscv_vmsgtu_vx_u8m1_b8(data, 'A' - 1, vl), __riscv_vmsleu_vx_u8m1_b8(data, 'Z', vl), vl);
    vbool8_t is_num = __riscv_vmand_mm_b8(__riscv_vmsgtu_vx_u8m1_b8(data, '0' - 1, vl), __riscv_vmsleu_vx_u8m1_b8(data, '9', vl), vl);
    vbool8_t is_minus = __riscv_vmseq_vx_u8m1_b8(data, '-', vl);
    vbool8_t is_dot = __riscv_vmseq_vx_u8m1_b8(data, '.', vl);
    vbool8_t is_underscore = __riscv_vmseq_vx_u8m1_b8(data, '_', vl);
    vbool8_t is_tilde = __riscv_vmseq_vx_u8m1_b8(data, '~', vl);
    vbool8_t is_safe = __riscv_vmor_mm_b8(is_az, is_AZ, vl);
    is_safe = __riscv_vmor_mm_b8(is_safe, is_num, vl);
    is_safe = __riscv_vmor_mm_b8(is_safe, is_minus, vl);
    is_safe = __riscv_vmor_mm_b8(is_safe, is_dot, vl);
    is_safe = __riscv_vmor_mm_b8(is_safe, is_underscore, vl);
    is_safe = __riscv_vmor_mm_b8(is_safe, is_tilde, vl);
    return is_safe;
}

Str url_encode_rvv(const char *str, size_t len, struct Arena *__restrict a) {
    if (!str || len == 0) return (Str){NULL, 0};
    size_t buf_size = len * 3 + 1;
    char *buf = arena_alloc(a, buf_size);
    if (!buf) return (Str){NULL, 0};
    char *p = buf;
    const unsigned char *us = (const unsigned char *)str;
    const unsigned char *end = us + len;
    while (us < end) {
        ptrdiff_t rem = end - us;
        if (rem < 32) {
            while (us < end) {
                unsigned char c = *us++;
                if (is_url_safe[c]) {
                    *p++ = (char)c;
                } else {
                    *p++ = '%';
                    *p++ = hex_digits[c >> 4];
                    *p++ = hex_digits[c & 0xF];
                }
            }
            break;
        }
        size_t vl = __riscv_vsetvl_e8m1(32);
        vuint8m1_t data = __riscv_vle8_v_u8m1(us, vl);
        vbool8_t is_safe = get_is_safe_rvv(data, vl);
        vbool8_t non_safe = __riscv_vmnot_m_b8(is_safe, vl);
        if (__riscv_vpopc_m_b8(non_safe, vl) == 0) {
            __riscv_vse8_v_u8m1((uint8_t *)p, data, vl);
            p += vl;
            us += vl;
        } else {
            for (size_t k = 0; k < vl; k++) {
                unsigned char c = us[k];
                if (is_url_safe[c]) {
                    *p++ = (char)c;
                } else {
                    *p++ = '%';
                    *p++ = hex_digits[c >> 4];
                    *p++ = hex_digits[c & 0xF];
                }
            }
            us += vl;
        }
    }
    return (Str){buf, (size_t)(p - buf)};
}

Str to_lower_rvv(const char *str, size_t len, struct Arena *__restrict a) {
    if (!str || len == 0) return (Str){NULL, 0};
    char *lower = arena_alloc(a, len + 1);
    if (!lower) return (Str){NULL, 0};
    size_t i = 0;
    for (; i + 32 <= len; i += 32) {
        size_t vl = __riscv_vsetvl_e8m1(32);
        vuint8m1_t chars = __riscv_vle8_v_u8m1((uint8_t *)(str + i), vl);
        vbool8_t is_upper = __riscv_vmand_mm_b8(__riscv_vmsgtu_vx_u8m1_b8(chars, 'A' - 1, vl), __riscv_vmsleu_vx_u8m1_b8(chars, 'Z', vl), vl);
        vuint8m1_t zero = __riscv_vmv_v_x_u8m1(0, vl);
        vuint8m1_t thirty_two = __riscv_vmv_v_x_u8m1(32, vl);
        vuint8m1_t add = __riscv_vmerge_vvm_u8m1(zero, thirty_two, is_upper, vl);
        vuint8m1_t lowered = __riscv_vadd_vv_u8m1(chars, add, vl);
        __riscv_vse8_v_u8m1((uint8_t *)(lower + i), lowered, vl);
    }
    for (; i < len; i++) {
        char c = str[i];
        lower[i] = (c >= 'A' && c <= 'Z') ? c + 32 : c;
    }
    return (Str){lower, len};
}
#endif

static uint64_t parse_eight_digits_swar(const char *chars) {
    uint64_t val;
    memcpy(&val, chars, 8);
    val = __builtin_bswap64(val);
    val -= 0x3030303030303030ULL;
    val = (val * 10) + (val >> 8);
    val = (val & 0x00ff00ff00ff00ffULL) * 100;
    val >>= 16;
    val = (val & 0x00ff00ff00ff00ffULL) * 100;
    val >>= 16;
    return ((val & 0x00ff00ff00ff00ffULL) * 100) >> 16;
}

static FORCE_INLINE int64_t fast_atol(const char *start, const char *end) {
    int64_t val = 0;
    int sign = 1;
    const char *p = start;
    if (p < end && *p == '-') {
        sign = -1;
        p++;
    }
    size_t len = end - p;
    while (len >= 8) {
        val = val * 100000000 + parse_eight_digits_swar(p);
        p += 8;
        len -= 8;
    }
    while (p < end && is_digit_lut[(unsigned char)*p]) {
        val = val * 10 + (*p - '0');
        p++;
    }
    return sign * val;
}

void (*skip_whitespace_fn)(char **, const char *);
void (*find_quote_fn)(char **, const char *);
void (*skip_digits_fn)(char **, const char *);
void (*skip_post_object_fn)(char **, const char *);
Str (*url_encode_fn)(const char *, size_t, struct Arena *__restrict);
Str (*to_lower_fn)(const char *, size_t, struct Arena *__restrict);

void extract_all(char *start, const char *end_pos, struct TorrentInfo *info) {
    memset(info, 0, sizeof(*info));
    char *p = start + 1;
    while (p < end_pos) {
        skip_whitespace_fn(&p, end_pos);
        if (UNLIKELY(p >= end_pos || *p != '"')) break;
        p++;
        const char *key_start = p;
        find_quote_fn(&p, end_pos);
        if (UNLIKELY(p >= end_pos)) break;
        size_t key_len = p - key_start;
        p++;
        skip_whitespace_fn(&p, end_pos);
        if (UNLIKELY(p >= end_pos || *p != ':')) break;
        p++;
        skip_whitespace_fn(&p, end_pos);
        char *val_start = p;
        int is_str = (*p == '"');
        size_t val_len = 0;
        if (is_str) {
            p++;
            val_start = p;
            find_quote_fn(&p, end_pos);
            if (UNLIKELY(p >= end_pos)) break;
            val_len = p - val_start;
            p++;
        } else {
            if (*p == '-') p++;
            skip_digits_fn(&p, end_pos);
            val_len = p - val_start;
        }
        int64_t val_l = 0;
        int val_i = 0;
        char *s = NULL;
        size_t slen = 0;
        if (is_str) {
            s = val_start;
            slen = val_len;
        } else {
            val_l = fast_atol(val_start, val_start + val_len);
            val_i = (int)val_l;
        }
        if (key_len == 2 && memcmp(key_start, "id", 2) == 0) {
            info->id = val_l;
        } else if (key_len == 4) {
            if (memcmp(key_start, "name", 4) == 0) {
                info->name = s;
                info->name_len = slen;
            } else if (memcmp(key_start, "size", 4) == 0) {
                info->size = val_l;
            } else if (memcmp(key_start, "anon", 4) == 0) {
                info->anon = val_i;
            } else if (memcmp(key_start, "imdb", 4) == 0) {
                info->imdb = s;
                info->imdb_len = slen;
            }
        } else if (key_len == 5 && memcmp(key_start, "added", 5) == 0) {
            info->added = (time_t)val_l;
        } else if (key_len == 6 && memcmp(key_start, "status", 6) == 0) {
            info->status = s;
            info->status_len = slen;
        } else if (key_len == 7 && memcmp(key_start, "seeders", 7) == 0) {
            info->seeders = val_i;
        } else if (key_len == 8) {
            if (memcmp(key_start, "category", 8) == 0) {
                info->category = val_i;
            } else if (memcmp(key_start, "leechers", 8) == 0) {
                info->leechers = val_i;
            } else if (memcmp(key_start, "username", 8) == 0) {
                info->username = s;
                info->username_len = slen;
            }
        } else if (key_len == 9) {
            if (memcmp(key_start, "info_hash", 9) == 0) {
                info->info_hash = s;
                info->info_hash_len = slen;
            } else if (memcmp(key_start, "num_files", 9) == 0) {
                info->num_files = val_i;
            }
        }
        skip_whitespace_fn(&p, end_pos);
        if (p < end_pos && *p == ',') p++;
        else break;
    }
}

char *read_file(const char *filename, long *len_out) {
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(len + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    fread(buf, 1, len, f);
    buf[len] = 0;
    fclose(f);
    *len_out = len;
    return buf;
}

static const char *days[7] = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};
static const char *months[12] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
static const char two_digits[] = "00010203040506070809101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899";

static const int days_in_month[2][12] = {
    {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
    {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
};

static int is_leap(int year) {
    return (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0));
}

static void fast_gmtime(time_t time, struct tm *tm) {
    time_t days = time / 86400;
    time_t secs = time % 86400;
    tm->tm_sec = secs % 60;
    secs /= 60;
    tm->tm_min = secs % 60;
    tm->tm_hour = secs / 60;
    tm->tm_wday = (days + 4) % 7;
    int year = 1970;
    while (1) {
        int ly = is_leap(year);
        time_t ydays = ly ? 366 : 365;
        if (days < ydays) break;
        days -= ydays;
        year++;
    }
    tm->tm_year = year - 1900;
    tm->tm_yday = days;
    int mon = 0;
    int leap = is_leap(year);
    while (1) {
        int d = days_in_month[leap][mon];
        if (days < d) break;
        days -= d;
        mon++;
    }
    tm->tm_mon = mon;
    tm->tm_mday = days + 1;
}

static FORCE_INLINE char *append_two(char *p, int n) {
    memcpy(p, &two_digits[n * 2], 2);
    return p + 2;
}

static FORCE_INLINE char *append_four(char *p, int n) {
    p = append_two(p, n / 100);
    p = append_two(p, n % 100);
    return p;
}

static FORCE_INLINE char *write_int64(char *p, int64_t val) {
    if (val == 0) {
        *p++ = '0';
        return p;
    }
    if (val < 0) {
        *p++ = '-';
        val = -val;
    }
    if (val < 10) {
        *p++ = '0' + val;
        return p;
    }
    if (val < 100) {
        p = append_two(p, (int)val);
        return p;
    }
    if (val < 10000) {
        p = append_four(p, (int)val);
        return p;
    }
    char tmp[20];
    int i = 0;
    while (val > 0) {
        tmp[i++] = '0' + (val % 10);
        val /= 10;
    }
    while (i > 0) {
        *p++ = tmp[--i];
    }
    return p;
}

static FORCE_INLINE char *write_int(char *p, int val) {
    return write_int64(p, val);
}

#define OUT_BUF_SIZE (16 << 20)
static char out_buf[OUT_BUF_SIZE];
static char *out_p;

void process_object(char *obj_start, const char *obj_end, int benchmark, struct Arena *__restrict a) {
    struct TorrentInfo info;
    extract_all(obj_start, obj_end, &info);
    if (UNLIKELY(!info.info_hash || !info.name)) {
        return;
    }
    Str ih_lower = to_lower_fn(info.info_hash, info.info_hash_len, a);
    Str dn = url_encode_fn(info.name, info.name_len, a);
    if (UNLIKELY(!ih_lower.data || !dn.data)) {
        return;
    }
    char magnet[2048];
    char *m = magnet;
    m = mempcpy_inline(m, "magnet:?xt=urn:btih:", 20);
    m = mempcpy_inline(m, ih_lower.data, ih_lower.len);
    m = mempcpy_inline(m, "&dn=", 4);
    m = mempcpy_inline(m, dn.data, dn.len);
    *m = 0;
    size_t magnet_len = m - magnet;

    struct tm tm;
    fast_gmtime(info.added, &tm);
    char pubdate[64];
    char *pd = pubdate;
    memcpy(pd, days[tm.tm_wday], 3);
    pd += 3;
    *pd++ = ',';
    *pd++ = ' ';
    pd = append_two(pd, tm.tm_mday);
    *pd++ = ' ';
    memcpy(pd, months[tm.tm_mon], 3);
    pd += 3;
    *pd++ = ' ';
    pd = append_four(pd, tm.tm_year + 1900);
    *pd++ = ' ';
    pd = append_two(pd, tm.tm_hour);
    *pd++ = ':';
    pd = append_two(pd, tm.tm_min);
    *pd++ = ':';
    pd = append_two(pd, tm.tm_sec);
    memcpy(pd, " +0000", 6);
    pd += 6;
    *pd = 0;
    size_t pubdate_len = pd - pubdate;

    if (!benchmark) {
        char *op = out_p;

        memcpy(op, "<item>\n  <title>", 16);
        op += 16;
        memcpy(op, info.name, info.name_len);
        op += info.name_len;
        memcpy(op, "</title>\n  <link>", 17);
        op += 17;
        memcpy(op, magnet, magnet_len);
        op += magnet_len;
        memcpy(op, "</link>\n  <pubDate>", 19);
        op += 19;
        memcpy(op, pubdate, pubdate_len);
        op += pubdate_len;
        memcpy(op, "</pubDate>\n  <guid isPermaLink=\"false\">", 39);
        op += 39;
        op = write_int64(op, info.id);
        memcpy(op, "</guid>\n  <description>", 23);
        op += 23;
        memcpy(op, info.name, info.name_len);
        op += info.name_len;
        memcpy(op, "</description>\n  <enclosure url=\"", 33);
        op += 33;
        memcpy(op, magnet, magnet_len);
        op += magnet_len;
        memcpy(op, "\" length=\"", 10);
        op += 10;
        op = write_int64(op, info.size);
        memcpy(op, "\" type=\"application/x-bittorrent\" />\n  <torznab:attr name=\"category\" value=\"", 75);
        op += 75;
        op = write_int(op, info.category);
        memcpy(op, "\" />\n  <torznab:attr name=\"size\" value=\"", 41);
        op += 41;
        op = write_int64(op, info.size);
        memcpy(op, "\" />\n  <torznab:attr name=\"files\" value=\"", 42);
        op += 42;
        op = write_int(op, info.num_files);
        memcpy(op, "\" />\n  <torznab:attr name=\"grabs\" value=\"0\" />\n  <torznab:attr name=\"seeders\" value=\"", 88);
        op += 88;
        op = write_int(op, info.seeders);
        memcpy(op, "\" />\n  <torznab:attr name=\"leechers\" value=\"", 45);
        op += 45;
        op = write_int(op, info.leechers);
        memcpy(op, "\" />\n  <torznab:attr name=\"peers\" value=\"", 42);
        op += 42;
        op = write_int(op, info.seeders + info.leechers);
        memcpy(op, "\" />\n  <torznab:attr name=\"infohash\" value=\"", 45);
        op += 45;
        memcpy(op, ih_lower.data, ih_lower.len);
        op += ih_lower.len;
        memcpy(op, "\" />\n  <torznab:attr name=\"magneturl\" value=\"", 46);
        op += 46;
        memcpy(op, magnet, magnet_len);
        op += magnet_len;
        memcpy(op, "\" />\n  <torznab:attr name=\"poster\" value=\"", 43);
        op += 43;
        memcpy(op, info.username, info.username_len);
        op += info.username_len;
        memcpy(op, "\" />\n", 5);
        op += 5;

        if (info.anon) {
            memcpy(op, "  <torznab:attr name=\"tag\" value=\"anonymous\" />\n", 49);
            op += 49;
        }
        if (info.status && info.status_len == 3 && memcmp(info.status, "vip", 3) == 0) {
            memcpy(op, "  <torznab:attr name=\"tag\" value=\"vip\" />\n", 43);
            op += 43;
        }
        if (info.imdb) {
            memcpy(op, "  <torznab:attr name=\"imdb\" value=\"", 35);
            op += 35;
            memcpy(op, info.imdb, info.imdb_len);
            op += info.imdb_len;
            memcpy(op, "\" />\n", 5);
            op += 5;
        }

        memcpy(op, "</item>\n", 8);
        op += 8;
        out_p = op;
    }
}

void *alloc_arena_memory(size_t size) {
#ifdef _WIN32
    return VirtualAlloc(NULL, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) return NULL;
    return ptr;
#endif
}

void free_arena_memory(void *base, size_t size) {
#ifdef _WIN32
    (void)size;
    VirtualFree(base, 0, MEM_RELEASE);
#else
    munmap(base, size);
#endif
}

int main(int argc, char *argv[]) {
#if defined(HAS_X86)
    cpu_level = get_cpu_level();
#endif
#if defined(__linux__) && defined(__aarch64__)
    has_sve = getauxval(AT_HWCAP) & HWCAP_SVE;
#endif
    skip_whitespace_fn = skip_whitespace_scalar;
    find_quote_fn = find_quote_scalar;
    skip_digits_fn = skip_digits_scalar;
    skip_post_object_fn = skip_post_object_scalar;
    url_encode_fn = url_encode_scalar;
    to_lower_fn = to_lower_scalar;
#if defined(HAS_X86)
    if (cpu_level == CPU_AVX512) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
        skip_whitespace_fn = skip_whitespace_avx512;
        find_quote_fn = find_quote_avx512;
        skip_digits_fn = skip_digits_avx512;
        skip_post_object_fn = skip_post_object_avx512;
        url_encode_fn = url_encode_avx512;
        to_lower_fn = to_lower_avx512;
#else
        cpu_level = CPU_AVX2;
        skip_whitespace_fn = skip_whitespace_avx2;
        find_quote_fn = find_quote_avx2;
        skip_digits_fn = skip_digits_avx2;
        skip_post_object_fn = skip_post_object_avx2;
        url_encode_fn = url_encode_avx2;
        to_lower_fn = to_lower_avx2;
#endif
    } else if (cpu_level == CPU_AVX2) {
        skip_whitespace_fn = skip_whitespace_avx2;
        find_quote_fn = find_quote_avx2;
        skip_digits_fn = skip_digits_avx2;
        skip_post_object_fn = skip_post_object_avx2;
        url_encode_fn = url_encode_avx2;
        to_lower_fn = to_lower_avx2;
    } else if (cpu_level >= CPU_SSE2) {
        skip_whitespace_fn = skip_whitespace_sse2;
        find_quote_fn = find_quote_sse2;
        skip_digits_fn = skip_digits_sse2;
        skip_post_object_fn = skip_post_object_sse2;
        url_encode_fn = url_encode_sse2;
        to_lower_fn = to_lower_sse2;
    }
#elif defined(HAS_ARM)
    if (has_sve) {
#if defined(__ARM_FEATURE_SVE)
        skip_whitespace_fn = skip_whitespace_sve;
        find_quote_fn = find_quote_sve;
        skip_digits_fn = skip_digits_sve;
        skip_post_object_fn = skip_post_object_sve;
        url_encode_fn = url_encode_sve;
        to_lower_fn = to_lower_sve;
#else
        has_sve = false;
        skip_whitespace_fn = skip_whitespace_neon;
        find_quote_fn = find_quote_neon;
        skip_digits_fn = skip_digits_neon;
        skip_post_object_fn = skip_post_object_neon;
        url_encode_fn = url_encode_neon;
        to_lower_fn = to_lower_neon;
#endif
    } else {
        skip_whitespace_fn = skip_whitespace_neon;
        find_quote_fn = find_quote_neon;
        skip_digits_fn = skip_digits_neon;
        skip_post_object_fn = skip_post_object_neon;
        url_encode_fn = url_encode_neon;
        to_lower_fn = to_lower_neon;
    }
#elif defined(HAS_POWER)
    if (__builtin_cpu_supports("vsx")) {
        skip_whitespace_fn = skip_whitespace_vsx;
        find_quote_fn = find_quote_vsx;
        skip_digits_fn = skip_digits_vsx;
        skip_post_object_fn = skip_post_object_vsx;
        url_encode_fn = url_encode_vsx;
        to_lower_fn = to_lower_vsx;
    }
#elif defined(HAS_RISCV)
    skip_whitespace_fn = skip_whitespace_rvv;
    find_quote_fn = find_quote_rvv;
    skip_digits_fn = skip_digits_rvv;
    skip_post_object_fn = skip_post_object_rvv;
    url_encode_fn = url_encode_rvv;
    to_lower_fn = to_lower_rvv;
#endif
    int benchmark = 0;
    if (argc > 1 && strcmp(argv[1], "-b") == 0) {
        benchmark = 1;
    }

    long json_len;
    char *json = read_file("../test.json", &json_len);
    if (!json) {
        fprintf(stderr, "Failed to read ../test.json\n");
        return 1;
    }

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    out_p = out_buf;
    if (!benchmark) {
        memcpy(out_p, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<rss version=\"2.0\" xmlns:atom=\"http://www.w3.org/2005/Atom\" xmlns:torznab=\"http://torznab.com/schemas/2015/feed\">\n<channel>\n  <atom:link href=\"https://example.com/api\" rel=\"self\" type=\"application/rss+xml\"/>\n  <title>Example Torznab Feed</title>\n  <description>Converted from JSON</description>\n", 319);
        out_p += 319;
    }

    char *p = json;
    char *json_end = json + json_len;
    p = memchr(p, '[', json_end - p);
    if (!p) {
        free(json);
        return 1;
    }
    p++;
    struct Arena a;
    size_t arena_size = json_len * 2;
    char *arena_buf = alloc_arena_memory(arena_size);
    if (!arena_buf) {
        free(json);
        return 1;
    }
    a.base = arena_buf;
    a.cur = arena_buf;
    a.end = arena_buf + arena_size;

    while (1) {
        char *obj_start = memchr(p, '{', json_end - p);
        if (!obj_start) break;
        p = obj_start;
        int level = 1;
        char *q = p + 1;
        while (q < json_end && level > 0) {
            if (*q == '{') level++;
            else if (*q == '}') level--;
            q++;
        }
        if (level > 0) break;
        process_object(p, q, benchmark, &a);
        p = q;
        skip_post_object_fn(&p, json_end);
        if (p >= json_end || *p == ']') break;
    }

    if (!benchmark) {
        memcpy(out_p, "</channel>\n</rss>\n", 18);
        out_p += 18;
        fwrite(out_buf, out_p - out_buf, 1, stdout);
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    long time_ns = (ts_end.tv_sec - ts_start.tv_sec) * 1000000000L + (ts_end.tv_nsec - ts_start.tv_nsec);
    if (benchmark) {
        printf("Time: %ld ns\n", time_ns);
    }

    free_arena_memory(arena_buf, arena_size);
    free(json);
    return 0;
}