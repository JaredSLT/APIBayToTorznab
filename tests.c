// tests.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <assert.h>

char *my_strndup(const char *s, size_t n) {
    size_t len = 0;
    while (len < n && s[len]) len++;
    char *p = malloc(len + 1);
    if (p) {
        memcpy(p, s, len);
        p[len] = 0;
    }
    return p;
}

char *extract_string(const char *obj, const char *key) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\":\"", key);
    char *pos = strstr(obj, search);
    if (!pos) return NULL;
    pos += strlen(search);
    char *end = strchr(pos, '"');
    if (!end) return NULL;
    return my_strndup(pos, end - pos);
}

int64_t extract_int64(const char *obj, const char *key) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\":", key);
    char *pos = strstr(obj, search);
    if (!pos) return -1;
    pos += strlen(search);
    while (*pos == ' ' || *pos == '\n' || *pos == '\r' || *pos == '\t') pos++;
    return strtoll(pos, NULL, 10);
}

char *url_encode(const char *str) {
    char *buf = malloc(strlen(str) * 3 + 1);
    if (!buf) return NULL;
    char *p = buf;
    while (*str) {
        if (isalnum((unsigned char)*str) || *str == '-' || *str == '_' || *str == '.' || *str == '~') {
            *p++ = *str;
        } else {
            sprintf(p, "%%%02X", (unsigned char)*str);
            p += 3;
        }
        str++;
    }
    *p = 0;
    return buf;
}

char *to_lower_str(const char *str) {
    char *lower = strdup(str);
    if (!lower) return NULL;
    char *p = lower;
    while (*p) {
        *p = (char)tolower((unsigned char)*p);
        p++;
    }
    return lower;
}

void test_extraction() {
    const char *test_obj = "{\"id\":80996803,\"info_hash\":\"DBB3FEC49D40EE29CC18B65236FCBDB7DEF443E5\",\"category\":208,\"name\":\"The Witcher S04E01 1080p WEB h264-ETHEL\",\"status\":\"vip\",\"num_files\":0,\"size\":2340892918,\"seeders\":6207,\"leechers\":1664,\"username\":\"jajaja\",\"added\":1761808802,\"anon\":0,\"imdb\":\"tt5180504\"}";

    int64_t id = extract_int64(test_obj, "id");
    assert(id == 80996803);

    char *info_hash = extract_string(test_obj, "info_hash");
    assert(strcmp(info_hash, "DBB3FEC49D40EE29CC18B65236FCBDB7DEF443E5") == 0);
    free(info_hash);

    int category = (int)extract_int64(test_obj, "category");
    assert(category == 208);

    char *name = extract_string(test_obj, "name");
    assert(strcmp(name, "The Witcher S04E01 1080p WEB h264-ETHEL") == 0);

    char *dn = url_encode(name);
    assert(strstr(dn, "%20") != NULL);
    free(dn);
    free(name);

    char *ih_lower = to_lower_str("DBB3FEC49D40EE29CC18B65236FCBDB7DEF443E5");
    assert(strcmp(ih_lower, "dbb3fec49d40ee29cc18b65236fcbdb7def443e5") == 0);
    free(ih_lower);

    char *imdb = extract_string(test_obj, "imdb");
    assert(strcmp(imdb, "tt5180504") == 0);
    free(imdb);

    char *missing = extract_string(test_obj, "nonexistent");
    assert(missing == NULL);
}

int main() {
    test_extraction();
    printf("All tests passed!\n");
    return 0;
}