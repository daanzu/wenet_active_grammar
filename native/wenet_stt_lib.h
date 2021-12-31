//
// This file is part of wenet_active_grammar.
// (c) Copyright 2021 by David Zurow
// Licensed under the AGPL-3.0; see LICENSE file.
//

#pragma once

#if defined(_MSC_VER)
    #ifdef WENET_STT_EXPORTS
        #define WENET_STT_API extern "C" __declspec(dllexport)
    #else
        #define WENET_STT_API extern "C" __declspec(dllimport)
    #endif
#elif defined(__GNUC__)
    // unnecessary
    #define WENET_STT_API extern "C" __attribute__((visibility("default")))
#else
    #define WENET_STT_API
    #pragma warning Unknown dynamic link import / export semantics.
#endif

#include <cstdint>

WENET_STT_API void *wenet_stt__construct_model(const char *config_json_cstr);
WENET_STT_API bool wenet_stt__destruct_model(void *model_vp);
WENET_STT_API bool wenet_stt__decode_utterance(void *model_vp, float *wav_samples, int32_t wav_samples_len, char *text, int32_t text_max_len);

WENET_STT_API void *wenet_stt__construct_decoder(void *model_vp);
WENET_STT_API bool wenet_stt__destruct_decoder(void *decoder_vp);
WENET_STT_API bool wenet_stt__decode(void *decoder_vp, float *wav_samples, int32_t wav_samples_len, bool finalize);
WENET_STT_API bool wenet_stt__get_result(void *decoder_vp, char *text, int32_t text_max_len, bool *final_p);
WENET_STT_API bool wenet_stt__reset(void *decoder_vp);

WENET_STT_API void *wenet_ag__construct_decoder(void *model_vp);
WENET_STT_API bool wenet_ag__destruct_decoder(void *decoder_vp);
WENET_STT_API bool wenet_ag__decode(void *decoder_vp, float *wav_samples, int32_t wav_samples_len, bool finalize);
WENET_STT_API bool wenet_ag__get_result(void *decoder_vp, char *text, int32_t text_max_len, bool *final_p, int32_t *rule_number_p);
WENET_STT_API bool wenet_ag__reset(void *decoder_vp);
WENET_STT_API bool wenet_ag__set_grammars_activity(void *decoder_vp, bool *grammars_activity_cp, int32_t grammars_activity_cp_size);
WENET_STT_API int32_t wenet_ag__add_grammar_fst(void *decoder_vp, void *grammar_fst_vp);
WENET_STT_API bool wenet_ag__reload_grammar_fst(void *decoder_vp, int32_t grammar_fst_index, void *grammar_fst_vp);
WENET_STT_API bool wenet_ag__remove_grammar_fst(void *decoder_vp, int32_t grammar_fst_index);

WENET_STT_API bool fst__init(int32_t eps_like_ilabels_len, int32_t eps_like_ilabels_cp[], int32_t silent_olabels_len, int32_t silent_olabels_cp[], int32_t wildcard_olabels_len, int32_t wildcard_olabels_cp[]);
WENET_STT_API void* fst__construct();
WENET_STT_API bool fst__destruct(void* fst_vp);
WENET_STT_API int32_t fst__add_state(void* fst_vp, float weight, bool initial);
WENET_STT_API bool fst__add_arc(void* fst_vp, int32_t src_state_id, int32_t dst_state_id, int32_t ilabel, int32_t olabel, float weight);
WENET_STT_API bool fst__compute_md5(void* fst_vp, char* md5_cp, char* dependencies_seed_md5_cp);
WENET_STT_API bool fst__has_path(void* fst_vp);
WENET_STT_API bool fst__has_eps_path(void* fst_vp, int32_t path_src_state, int32_t path_dst_state);
WENET_STT_API bool fst__does_match(void* fst_vp, int32_t target_labels_len, int32_t target_labels_cp[], int32_t output_labels_cp[], int32_t* output_labels_len);
WENET_STT_API void* fst__load_file(char* filename_cp);
WENET_STT_API bool fst__write_file(void* fst_vp, char* filename_cp);
WENET_STT_API bool fst__write_file_const(void* fst_vp, char* filename_cp);
WENET_STT_API bool fst__print(void* fst_vp, char* filename_cp);
WENET_STT_API void* fst__compile_text(char* fst_text_cp, char* isymbols_file_cp, char* osymbols_file_cp);
