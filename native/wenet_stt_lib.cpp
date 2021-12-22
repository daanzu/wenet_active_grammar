//
// This file is part of wenet_active_grammar.
// (c) Copyright 2021 by David Zurow
// Licensed under the AGPL-3.0; see LICENSE file.
//

#include <torch/script.h>

#include "decoder/params.h"
#include "utils/log.h"
#include "utils/timer.h"
#include "utils/utils.h"

#include "nlohmann/json.hpp"

#define BEGIN_INTERFACE_CATCH_HANDLER \
    try {
#define END_INTERFACE_CATCH_HANDLER(expr) \
    } catch(const std::exception& e) { \
        std::cerr << "Trying to survive fatal exception: " << e.what(); \
        return (expr); \
    }

namespace wenet {
    void from_json(const nlohmann::json& j, CtcEndpointConfig& c) {
        if (j.contains("blank")) j.at("blank").get_to(c.blank);
        if (j.contains("blank_threshold")) j.at("blank_threshold").get_to(c.blank_threshold);
        // if (j.contains("rule1")) j.at("rule1").get_to(c.rule1);
        // if (j.contains("rule2")) j.at("rule2").get_to(c.rule2);
        // if (j.contains("rule3")) j.at("rule3").get_to(c.rule3);
    }
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CtcEndpointRule,
        must_decoded_sth,
        min_trailing_silence,
        min_utterance_length
    );
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CtcPrefixBeamSearchOptions,
        blank,
        first_beam_size,
        second_beam_size
    );
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CtcWfstBeamSearchOptions,
        max_active,
        min_active,
        beam,
        lattice_beam,
        acoustic_scale,
        nbest,
        blank_skip_thresh
    );
    // NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DecodeResource);
}


std::shared_ptr<wenet::FeaturePipelineConfig> InitFeaturePipelineConfigFromJson(const nlohmann::json& j) {
    return std::make_shared<wenet::FeaturePipelineConfig>(
        j.at("num_bins").get<int>(),
        j.at("sample_rate").get<int>()
    );
}

std::shared_ptr<wenet::FeaturePipelineConfig> InitFeaturePipelineConfigFromSimpleJson(const nlohmann::json& j) {
    auto num_bins = (j.contains("num_bins")) ? j.at("num_bins").get<int>() : FLAGS_num_bins;
    auto sample_rate = (j.contains("sample_rate")) ? j.at("sample_rate").get<int>() : FLAGS_sample_rate;
    return std::make_shared<wenet::FeaturePipelineConfig>(num_bins, sample_rate);
}

std::shared_ptr<wenet::DecodeOptions> InitDecodeOptionsFromJson(const nlohmann::json& j) {
    if (!j.is_object()) LOG(FATAL) << "decode_options must be a valid JSON object";
    auto decode_config = std::make_shared<wenet::DecodeOptions>();
    if (j.contains("chunk_size")) j.at("chunk_size").get_to(decode_config->chunk_size);
    if (j.contains("num_left_chunks")) j.at("num_left_chunks").get_to(decode_config->num_left_chunks);
    if (j.contains("ctc_weight")) j.at("ctc_weight").get_to(decode_config->ctc_weight);
    if (j.contains("rescoring_weight")) j.at("rescoring_weight").get_to(decode_config->rescoring_weight);
    if (j.contains("reverse_weight")) j.at("reverse_weight").get_to(decode_config->reverse_weight);
    if (j.contains("ctc_endpoint_config")) j.at("ctc_endpoint_config").get_to(decode_config->ctc_endpoint_config);
    if (j.contains("ctc_prefix_search_opts")) j.at("ctc_prefix_search_opts").get_to(decode_config->ctc_prefix_search_opts);
    if (j.contains("ctc_wfst_search_opts")) j.at("ctc_wfst_search_opts").get_to(decode_config->ctc_wfst_search_opts);
    return decode_config;
}

std::shared_ptr<wenet::DecodeOptions> InitDecodeOptionsFromSimpleJson(const nlohmann::json& j) {
    if (!j.is_object()) LOG(FATAL) << "decode_options must be a valid JSON object";
    auto decode_config = std::make_shared<wenet::DecodeOptions>();
    if (j.contains("chunk_size")) { j.at("chunk_size").get_to(decode_config->chunk_size); } else { decode_config->chunk_size = FLAGS_chunk_size; }
    if (j.contains("num_left_chunks")) { j.at("num_left_chunks").get_to(decode_config->num_left_chunks); } else { decode_config->num_left_chunks = FLAGS_num_left_chunks; }
    if (j.contains("ctc_weight")) { j.at("ctc_weight").get_to(decode_config->ctc_weight); } else { decode_config->ctc_weight = FLAGS_ctc_weight; }
    if (j.contains("rescoring_weight")) { j.at("rescoring_weight").get_to(decode_config->rescoring_weight); } else { decode_config->rescoring_weight = FLAGS_rescoring_weight; }
    if (j.contains("reverse_weight")) { j.at("reverse_weight").get_to(decode_config->reverse_weight); } else { decode_config->reverse_weight = FLAGS_reverse_weight; }
    if (j.contains("max_active")) { j.at("max_active").get_to(decode_config->ctc_wfst_search_opts.max_active); } else { decode_config->ctc_wfst_search_opts.max_active = FLAGS_max_active; }
    if (j.contains("min_active")) { j.at("min_active").get_to(decode_config->ctc_wfst_search_opts.min_active); } else { decode_config->ctc_wfst_search_opts.min_active = FLAGS_min_active; }
    if (j.contains("beam")) { j.at("beam").get_to(decode_config->ctc_wfst_search_opts.beam); } else { decode_config->ctc_wfst_search_opts.beam = FLAGS_beam; }
    if (j.contains("lattice_beam")) { j.at("lattice_beam").get_to(decode_config->ctc_wfst_search_opts.lattice_beam); } else { decode_config->ctc_wfst_search_opts.lattice_beam = FLAGS_lattice_beam; }
    if (j.contains("acoustic_scale")) { j.at("acoustic_scale").get_to(decode_config->ctc_wfst_search_opts.acoustic_scale); } else { decode_config->ctc_wfst_search_opts.acoustic_scale = FLAGS_acoustic_scale; }
    if (j.contains("blank_skip_thresh")) { j.at("blank_skip_thresh").get_to(decode_config->ctc_wfst_search_opts.blank_skip_thresh); } else { decode_config->ctc_wfst_search_opts.blank_skip_thresh = FLAGS_blank_skip_thresh; }
    if (j.contains("nbest")) { j.at("nbest").get_to(decode_config->ctc_wfst_search_opts.nbest); } else { decode_config->ctc_wfst_search_opts.nbest = FLAGS_nbest; }
    return decode_config;
}

std::shared_ptr<wenet::DecodeResource> InitDecodeResourceFromJson(const nlohmann::json& j) {
    if (!j.is_object()) LOG(FATAL) << "decode_resource must be a valid JSON object";
    auto resource = std::make_shared<wenet::DecodeResource>();

    auto model_path = j.at("model_path").get<std::string>();
    auto num_threads = j.at("num_threads").get<int>();
    LOG(INFO) << "Reading model " << model_path << " to use " << num_threads << " threads";
    auto model = std::make_shared<wenet::TorchAsrModel>();
    model->Read(model_path, num_threads);
    resource->model = model;

    std::shared_ptr<fst::Fst<fst::StdArc>> fst = nullptr;
    if (j.contains("fst_path")) {
        auto fst_path = j.at("fst_path").get<std::string>();
        LOG(INFO) << "Reading fst " << fst_path;
        fst.reset(fst::Fst<fst::StdArc>::Read(fst_path));
        CHECK(fst != nullptr);
    }
    resource->fst = fst;

    auto dict_path = j.at("dict_path").get<std::string>();
    LOG(INFO) << "Reading symbol table " << dict_path;
    auto symbol_table = std::shared_ptr<fst::SymbolTable>(
        fst::SymbolTable::ReadText(dict_path));
    resource->symbol_table = symbol_table;

    std::shared_ptr<fst::SymbolTable> unit_table = nullptr;
    if (j.contains("unit_path")) {
        auto unit_path = j.at("unit_path").get<std::string>();
        LOG(INFO) << "Reading unit table " << unit_path;
        unit_table = std::shared_ptr<fst::SymbolTable>(
            fst::SymbolTable::ReadText(unit_path));
        CHECK(unit_table != nullptr);
    } else if (fst == nullptr) {
        LOG(INFO) << "Using symbol table as unit table";
        unit_table = symbol_table;
    }
    resource->unit_table = unit_table;

    return resource;
}

std::shared_ptr<wenet::DecodeResource> InitDecodeResourceFromSimpleJson(const nlohmann::json& j) {
    if (!j.is_object()) LOG(FATAL) << "decode_resource must be a valid JSON object";
    auto resource = std::make_shared<wenet::DecodeResource>();

    auto model_path = j.at("model_path").get<std::string>();
    auto num_threads = (j.contains("num_threads")) ? j.at("num_threads").get<int>() : FLAGS_num_threads;
    LOG(INFO) << "Reading model " << model_path << " to use " << num_threads << " threads";
    auto model = std::make_shared<wenet::TorchAsrModel>();
    model->Read(model_path, num_threads);
    resource->model = model;

    std::shared_ptr<fst::Fst<fst::StdArc>> fst = nullptr;
    if (j.contains("fst_path")) {
        auto fst_path = j.at("fst_path").get<std::string>();
        LOG(INFO) << "Reading fst " << fst_path;
        fst.reset(fst::Fst<fst::StdArc>::Read(fst_path));
        CHECK(fst != nullptr);
    }
    resource->fst = fst;

    auto dict_path = j.at("dict_path").get<std::string>();
    LOG(INFO) << "Reading symbol table " << dict_path;
    auto symbol_table = std::shared_ptr<fst::SymbolTable>(
        fst::SymbolTable::ReadText(dict_path));
    resource->symbol_table = symbol_table;

    std::shared_ptr<fst::SymbolTable> unit_table = nullptr;
    if (j.contains("unit_path")) {
        auto unit_path = j.at("unit_path").get<std::string>();
        LOG(INFO) << "Reading unit table " << unit_path;
        unit_table = std::shared_ptr<fst::SymbolTable>(
            fst::SymbolTable::ReadText(unit_path));
        CHECK(unit_table != nullptr);
    } else if (fst == nullptr) {
        LOG(INFO) << "Using symbol table as unit table";
        unit_table = symbol_table;
    }
    resource->unit_table = unit_table;

    if (j.contains("grammar_symbol_path")) {
        auto grammar_symbol_path = j.at("grammar_symbol_path").get<std::string>();
        LOG(INFO) << "Reading grammar symbol table " << grammar_symbol_path;
        auto grammar_symbol_table = std::shared_ptr<fst::SymbolTable>(
            fst::SymbolTable::ReadText(grammar_symbol_path));
        CHECK(grammar_symbol_table != nullptr);
        resource->grammar_symbol_table = grammar_symbol_table;
        resource->unit_table = symbol_table;
    }

    // FIXME: handle context graph

    wenet::PostProcessOptions post_process_opts;
    post_process_opts.language_type = j.contains("language_type") ? static_cast<wenet::LanguageType>(j.at("language_type").get<int32_t>()) : wenet::kMandarinEnglish;
    post_process_opts.lowercase = j.contains("lowercase") ? j.at("lowercase").get<bool>() : true;
    resource->post_processor =
      std::make_shared<wenet::PostProcessor>(std::move(post_process_opts));

    return resource;
}

static bool one_time_initialized_ = false;

struct WenetSTTModel {
    std::string config_json_str_;
    nlohmann::json config_json_;
    std::shared_ptr<wenet::FeaturePipelineConfig> feature_config_;
    std::shared_ptr<wenet::DecodeOptions> decode_config_;
    std::shared_ptr<wenet::DecodeResource> decode_resource_;

    WenetSTTModel(const std::string& config_json_str) {
        if (!one_time_initialized_) {
            one_time_initialized_ = true;
            google::InitGoogleLogging("WenetSTT");
            LOG(INFO) << "Initializing WenetSTT in process " << getpid();
        }

        if (!config_json_str.empty()) {
            config_json_ = nlohmann::json::parse(config_json_str);
            if (!config_json_.is_object()) LOG(FATAL) << "config_json_str must be a valid JSON object";
            config_json_str_ = config_json_str;
            feature_config_ = InitFeaturePipelineConfigFromSimpleJson(config_json_);
            decode_config_ = InitDecodeOptionsFromSimpleJson(config_json_);
            decode_resource_ = InitDecodeResourceFromSimpleJson(config_json_);
        }
    }

    int sample_rate() const { return feature_config_->sample_rate; }
    bool is_streaming() const { return decode_config_->chunk_size > 0; }

    std::string DecodeUtterance(const std::vector<float>& wav_samples) {
        auto feature_pipeline = std::make_shared<wenet::FeaturePipeline>(*feature_config_);
        feature_pipeline->AcceptWaveform(wav_samples);
        feature_pipeline->set_input_finished();
        LOG(INFO) << "Num frames: " << feature_pipeline->num_frames();
        wenet::TorchAsrDecoder decoder(feature_pipeline, decode_resource_, *decode_config_);

        int wav_duration = wav_samples.size() / sample_rate();
        int decode_time = 0;
        while (true) {
            wenet::Timer timer;
            wenet::DecodeState state = decoder.Decode();
            if (state == wenet::DecodeState::kEndFeats) {
                decoder.Rescoring();
            }
            int chunk_decode_time = timer.Elapsed();
            decode_time += chunk_decode_time;
            if (decoder.DecodedSomething()) {
                LOG(INFO) << "Partial result: " << decoder.result()[0].sentence;
            }
            if (state == wenet::DecodeState::kEndFeats) {
                break;
            }
        }
        std::string hypothesis;
        if (decoder.DecodedSomething()) {
            hypothesis = decoder.result()[0].sentence;
        }
        LOG(INFO) << "Final result: " << hypothesis;
        LOG(INFO) << "Decoded " << wav_duration << "ms audio taking " << decode_time << "ms. RTF: " << std::setprecision(4) << static_cast<float>(decode_time) / wav_duration;

        // Strip any trailing whitespace
        auto last_pos = hypothesis.find_last_not_of(' ');
        hypothesis = hypothesis.substr(0, last_pos + 1);

        return hypothesis;
    }
};

class WenetAGDecoder;

class WenetSTTDecoder {
    friend class WenetAGDecoder;

public:
    WenetSTTDecoder(std::shared_ptr<const WenetSTTModel> model) :
        model_(model),
        feature_pipeline_(std::make_shared<wenet::FeaturePipeline>(*model_->feature_config_)),
        decoder_(std::make_shared<wenet::TorchAsrDecoder>(feature_pipeline_, model_->decode_resource_, *model_->decode_config_)),
        decode_thread_(std::make_unique<std::thread>(&WenetSTTDecoder::DecodeThreadFunc, this)) {}

    ~WenetSTTDecoder() {
        if (decode_thread_->joinable()) {
            if (!finalized_) {
                Finalize();
            }
            decode_thread_->join();
        }
    }

    // Decodes given audio block, and finalizes if passed true. Must not be called again after finalizing without having called Reset().
    void Decode(const std::vector<float>& wav_samples, bool finalize) {
        CHECK(!finalized_);
        started_ = true;
        if (!wav_samples.empty()) {
            feature_pipeline_->AcceptWaveform(wav_samples);
        }
        if (finalize) {
            Finalize();
        }
    }

    // Places current result into given string, and returns true if it was final.
    bool GetResult(std::string& result) {
        std::lock_guard<std::mutex> lock(result_mutex_);
        result = result_;
        return result_is_final_;
    }

    // Reset decoder for decoding a new utterance.
    void Reset() {
        if (decode_thread_->joinable()) {
            if (!finalized_) {
                Finalize();
            }
            decode_thread_->join();
        }

        started_ = false;
        finalized_ = false;
        result_.clear();
        result_is_final_ = false;
        feature_pipeline_->Reset();
        decoder_->Reset();
        CHECK(!decode_thread_->joinable());
        decode_thread_ = std::make_unique<std::thread>(&WenetSTTDecoder::DecodeThreadFunc, this);
    }

protected:

    void Finalize() {
        feature_pipeline_->set_input_finished();
        finalized_ = true;
    }

    // Decode in separate thread.
    void DecodeThreadFunc() {
        while (true) {
            wenet::DecodeState state = decoder_->Decode();
            if (!started_) {
                CHECK_EQ(state, wenet::DecodeState::kEndFeats);
                VLOG(2) << "Decoder not started yet, but was finalized, so terminating thread.";
                break;
            } else if (state == wenet::DecodeState::kEndFeats) {
                CHECK(finalized_);
                decoder_->Rescoring();
                if (decoder_->DecodedSomething()) {
                    VLOG(1) << "Final result: " << decoder_->result()[0].sentence;
                    std::lock_guard<std::mutex> lock(result_mutex_);
                    result_ = decoder_->result()[0].sentence;
                }
                result_is_final_ = true;
                break;
            } else if (state == wenet::DecodeState::kEndpoint) {
                CHECK(false) << "Endpoint reached";
                decoder_->ResetContinuousDecoding();
            } else {
                if (decoder_->DecodedSomething()) {
                    VLOG(1) << "Partial result: " << decoder_->result()[0].sentence;
                    std::lock_guard<std::mutex> lock(result_mutex_);
                    result_ = decoder_->result()[0].sentence;
                }
            }
        }
    }

    std::shared_ptr<const WenetSTTModel> model_;
    std::shared_ptr<wenet::FeaturePipeline> feature_pipeline_;
    std::shared_ptr<wenet::TorchAsrDecoder> decoder_;
    std::unique_ptr<std::thread> decode_thread_;

    bool started_ = false;
    bool finalized_ = false;

    std::mutex result_mutex_;
    std::string result_;
    bool result_is_final_ = false;
};

class WenetAGDecoder {
public:
    WenetAGDecoder(std::shared_ptr<const WenetSTTModel> model) {
        CHECK_NOTNULL(model->decode_resource_->grammar_symbol_table);
        word_syms_ = model->decode_resource_->grammar_symbol_table;
        rules_words_offset_ = word_syms_->Find(model->config_json_.at("rule0_label").get<std::string>());
        CHECK_NE(rules_words_offset_, fst::SymbolTable::kNoSymbol) << "Could not find rule0 in symbol table";
        nonterm_end_label_ = word_syms_->Find(model->config_json_.at("nonterm_end_label").get<std::string>());
        CHECK_NE(nonterm_end_label_, fst::SymbolTable::kNoSymbol) << "Could not find nonterm_end in symbol table";
        max_num_rules_ = model->config_json_.at("max_num_rules").get<int64>();
        if (model->config_json_.contains("skip_words")) {
            for (const auto& word : model->config_json_.at("skip_words")) {
                skip_word_ids_.emplace_back(word_syms_->Find(word.get<std::string>()));
            }
        }
        // CHECK_GT(model->decode_config_->chunk_size, 0);

        decode_fst_.reset(BuildDecodeFst());
        model->decode_resource_->fst = decode_fst_;
        // Guarantee we are using CtcPrefixWfstBeamSearch.
        CHECK_NOTNULL(model->decode_resource_->fst);
        CHECK_NOTNULL(model->decode_resource_->grammar_symbol_table);
        stt_decoder_ = std::make_unique<WenetSTTDecoder>(model);
    }

    ~WenetAGDecoder() {}

    void Decode(const std::vector<float>& wav_samples, bool finalize) {
        if (!decode_fst_) {
            Reset();
        }
        CHECK_NOTNULL(decode_fst_);
        stt_decoder_->Decode(wav_samples, finalize);
    }

    bool GetResult(std::string& result) { return stt_decoder_->GetResult(result); }

    void Reset() {
        if (!decode_fst_) {
            VLOG(1) << "Rebuilding decode_fst_";
            decode_fst_.reset(BuildDecodeFst());
            auto& searcher = stt_decoder_->decoder_->get_searcher();
            CHECK_EQ(searcher.Type(), wenet::SearchType::kPrefixWfstBeamSearch);
            auto& cast_searcher = static_cast<wenet::CtcPrefixWfstBeamSearch&>(searcher);
            cast_searcher.ResetFst(decode_fst_);
        }
        stt_decoder_->Reset();
    }

    void SetGrammarsActivity(const std::vector<bool>& grammars_activity) { grammars_activity_ = grammars_activity; }

    // Does not take ownership of FST!?
    int32 AddGrammarFst(fst::StdExpandedFst* grammar_fst, std::string grammar_name = "<unnamed>") {
        InvalidateDecodeFst();
        // ExecutionTimer timer("AddGrammarFst:loading");
        auto grammar_fst_index = grammar_fsts_.size();
        VLOG(1) << "adding FST #" << grammar_fst_index << " @ 0x" << grammar_fst << " " << grammar_name;
        grammar_fsts_.push_back(grammar_fst);
        // grammar_fsts_name_map_[grammar_fst] = grammar_name;
        return grammar_fst_index;
    }

    bool ReloadGrammarFst(int32 grammar_fst_index, fst::StdExpandedFst* grammar_fst, std::string grammar_name = "<unnamed>") {
        InvalidateDecodeFst();
        auto old_grammar_fst = grammar_fsts_.at(grammar_fst_index);
        // grammar_fsts_name_map_.erase(old_grammar_fst);
        VLOG(1) << "reloading FST #" << grammar_fst_index << " @ 0x" << grammar_fst << " " << grammar_name;
        grammar_fsts_.at(grammar_fst_index) = grammar_fst;
        // grammar_fsts_name_map_[grammar_fst] = grammar_name;
        return true;
    }

    bool RemoveGrammarFst(int32 grammar_fst_index) {
        InvalidateDecodeFst();
        auto grammar_fst = grammar_fsts_.at(grammar_fst_index);
        VLOG(1) << "removing FST #" << grammar_fst_index << " @ 0x" << grammar_fst;
        grammar_fsts_.erase(grammar_fsts_.begin() + grammar_fst_index);
        // grammar_fsts_name_map_.erase(grammar_fst);
        return true;
    }

protected:

    bool InvalidateDecodeFst() {
        CHECK(!stt_decoder_ || !stt_decoder_->started_) << "Cannot invalidate grammar FSTs in the middle of decoding";
        if (decode_fst_) {
            VLOG(1) << "Invalidating decode_fst_";
            decode_fst_.reset();
            return true;
        }
        return false;
    }

    fst::StdFst* BuildDecodeFst() {
        InvalidateDecodeFst();
        // ExecutionTimer timer("BuildDecodeFst", -1);

        std::vector<std::pair<int32, const fst::StdFst *> > label_fst_pairs;
        auto top_fst_nonterm = rules_words_offset_ + max_num_rules_;  // Give our on-demand-generated top-fst the label index just past the last rule (thus unused).

        // Build top_fst
        fst::StdVectorFst top_fst;
        auto start_state = top_fst.AddState();
        top_fst.SetStart(start_state);
        auto final_state = top_fst.AddState();
        top_fst.SetFinal(final_state, 0.0);

        top_fst.SetFinal(start_state, 0.0);  // Allow start state to be final, for no rule
        top_fst.AddArc(0, fst::StdArc(0, 0, 0.0, final_state));  // Allow epsilon transition to final state, for no rule
        for (auto skip_word_id : skip_word_ids_)
            top_fst.AddArc(0, fst::StdArc(skip_word_id, 0, 0.0, final_state));

        CHECK_LE(grammar_fsts_.size(), max_num_rules_);
        CHECK_EQ(grammars_activity_.size(), grammar_fsts_.size());
        for (size_t i = 0; i < grammar_fsts_.size(); ++i) {
            if (grammars_activity_.at(i)) {
                top_fst.AddArc(0, fst::StdArc(0, (rules_words_offset_ + i), 0.0, final_state));
                label_fst_pairs.emplace_back((rules_words_offset_ + i), grammar_fsts_.at(i));
            }
        }

        // if (dictation_fst_ != nullptr)
        //     label_fst_pairs.emplace_back(word_syms_->Find("#nonterm:dictation"), dictation_fst_);
        // top_fst.AddArc(0, StdArc(0, word_syms_->Find("#nonterm:dictation"), 0.0, final_state));

        fst::ArcSort(&top_fst, fst::StdILabelCompare());
        label_fst_pairs.emplace_back(top_fst_nonterm, new fst::StdConstFst(top_fst));
        // timer.step("top_fst");

        // Save to files for debugging.
        if (false) {
            top_fst.Write("top_fst.fst");
            for (const auto& label_fst_pair : label_fst_pairs) {
                label_fst_pair.second->Write("rule" + std::to_string(label_fst_pair.first) + ".fst");
            }
        }

        // Create replace FST, either on-demand or pre-compiled.
        fst::ReplaceFstOptions<fst::StdArc> replace_options(top_fst_nonterm, fst::REPLACE_LABEL_OUTPUT, fst::REPLACE_LABEL_OUTPUT, nonterm_end_label_);
        if (true) {
            // auto cache_size = 1ULL << 30;  // config_->decode_fst_cache_size
            // replace_options.gc_limit = cache_size;  // ReplaceFst needs the most cache space of the 3 delayed Fsts?
            return new fst::StdReplaceFst(label_fst_pairs, replace_options);
        } else {
            auto debug_replace_fst = new fst::StdVectorFst();
            fst::Replace(label_fst_pairs, debug_replace_fst, replace_options);
            // fst::RmEpsilon(debug_replace_fst);
            // debug_replace_fst->Write("debug_replace_fst.fst");
            return debug_replace_fst;
        }
    }

    int64 rules_words_offset_ = fst::kNoLabel;
    int64 nonterm_end_label_ = fst::kNoLabel;
    int64 max_num_rules_ = 0;
    std::vector<int64> skip_word_ids_;

    std::shared_ptr<fst::SymbolTable> word_syms_ = nullptr;
    std::vector<fst::StdFst*> grammar_fsts_;
    std::vector<bool> grammars_activity_;  // Bitfield of whether each grammar is active for current/upcoming utterance.
    std::shared_ptr<fst::StdFst> decode_fst_ = nullptr;
    std::unique_ptr<WenetSTTDecoder> stt_decoder_ = nullptr;
};


extern "C" {
#include "wenet_stt_lib.h"
}

void *wenet_stt__construct_model(const char *config_json_cstr) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto model = new WenetSTTModel(config_json_cstr);
    return model;
    END_INTERFACE_CATCH_HANDLER(nullptr)
}

bool wenet_stt__destruct_model(void *model_vp) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto model = static_cast<WenetSTTModel*>(model_vp);
    delete model;
    return true;
    END_INTERFACE_CATCH_HANDLER(false)
}

bool wenet_stt__decode_utterance(void *model_vp, float *wav_samples, int32_t wav_samples_len, char *text, int32_t text_max_len) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto model = static_cast<WenetSTTModel*>(model_vp);
    auto hypothesis = model->DecodeUtterance(std::vector<float>(wav_samples, wav_samples + wav_samples_len));
    auto cstr = hypothesis.c_str();
    strncpy(text, cstr, text_max_len);
    text[text_max_len - 1] = 0;  // Just in case.
    return true;
    END_INTERFACE_CATCH_HANDLER(false)
}


void *wenet_stt__construct_decoder(void *model_vp) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto model = static_cast<WenetSTTModel*>(model_vp);
    auto decoder = new WenetSTTDecoder(std::make_shared<const WenetSTTModel>(*model));
    return decoder;
    END_INTERFACE_CATCH_HANDLER(nullptr)
}

bool wenet_stt__destruct_decoder(void *decoder_vp) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto decoder = static_cast<WenetSTTDecoder*>(decoder_vp);
    delete decoder;
    return true;
    END_INTERFACE_CATCH_HANDLER(false)
}

bool wenet_stt__decode(void *decoder_vp, float *wav_samples, int32_t wav_samples_len, bool finalize) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto decoder = static_cast<WenetSTTDecoder*>(decoder_vp);
    decoder->Decode(std::vector<float>(wav_samples, wav_samples + wav_samples_len), finalize);
    return true;
    END_INTERFACE_CATCH_HANDLER(false)
}

bool wenet_stt__get_result(void *decoder_vp, char *text, int32_t text_max_len, bool *final_p) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto decoder = static_cast<WenetSTTDecoder*>(decoder_vp);
    std::string result;
    *final_p = decoder->GetResult(result);
    strncpy(text, result.c_str(), text_max_len);
    text[text_max_len - 1] = 0;  // Just in case.
    return true;
    END_INTERFACE_CATCH_HANDLER(false)
}

bool wenet_stt__reset(void *decoder_vp) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto decoder = static_cast<WenetSTTDecoder*>(decoder_vp);
    decoder->Reset();
    return true;
    END_INTERFACE_CATCH_HANDLER(false)
}


void *wenet_ag__construct_decoder(void *model_vp) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto model = static_cast<WenetSTTModel*>(model_vp);
    auto decoder = new WenetAGDecoder(std::make_shared<const WenetSTTModel>(*model));
    return decoder;
    END_INTERFACE_CATCH_HANDLER(nullptr)
}

bool wenet_ag__destruct_decoder(void *decoder_vp) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto decoder = static_cast<WenetAGDecoder*>(decoder_vp);
    delete decoder;
    return true;
    END_INTERFACE_CATCH_HANDLER(false)
}

bool wenet_ag__decode(void *decoder_vp, float *wav_samples, int32_t wav_samples_len, bool finalize) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto decoder = static_cast<WenetAGDecoder*>(decoder_vp);
    decoder->Decode(std::vector<float>(wav_samples, wav_samples + wav_samples_len), finalize);
    return true;
    END_INTERFACE_CATCH_HANDLER(false)
}

bool wenet_ag__get_result(void *decoder_vp, char *text, int32_t text_max_len, bool *final_p) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto decoder = static_cast<WenetAGDecoder*>(decoder_vp);
    std::string result;
    *final_p = decoder->GetResult(result);
    strncpy(text, result.c_str(), text_max_len);
    text[text_max_len - 1] = 0;  // Just in case.
    return true;
    END_INTERFACE_CATCH_HANDLER(false)
}

bool wenet_ag__reset(void *decoder_vp) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto decoder = static_cast<WenetAGDecoder*>(decoder_vp);
    decoder->Reset();
    return true;
    END_INTERFACE_CATCH_HANDLER(false)
}

bool wenet_ag__set_grammars_activity(void *decoder_vp, bool *grammars_activity_cp, int32_t grammars_activity_cp_size) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto decoder = static_cast<WenetAGDecoder*>(decoder_vp);
    std::vector<bool> grammars_activity(grammars_activity_cp, grammars_activity_cp + grammars_activity_cp_size);
    decoder->SetGrammarsActivity(grammars_activity);
    return true;
    END_INTERFACE_CATCH_HANDLER(false)
}

int32_t wenet_ag__add_grammar_fst(void *decoder_vp, void *grammar_fst_vp) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto decoder = static_cast<WenetAGDecoder*>(decoder_vp);
    auto grammar_fst = static_cast<fst::StdExpandedFst*>(grammar_fst_vp);
    auto result = decoder->AddGrammarFst(grammar_fst);
    return result;
    END_INTERFACE_CATCH_HANDLER(-1)
}

bool wenet_ag__reload_grammar_fst(void *decoder_vp, int32_t grammar_fst_index, void *grammar_fst_vp) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto decoder = static_cast<WenetAGDecoder*>(decoder_vp);
    auto grammar_fst = static_cast<fst::StdExpandedFst*>(grammar_fst_vp);
    auto result = decoder->ReloadGrammarFst(grammar_fst_index, grammar_fst);
    return true;
    END_INTERFACE_CATCH_HANDLER(false)
}

bool wenet_ag__remove_grammar_fst(void *decoder_vp, int32_t grammar_fst_index) {
    BEGIN_INTERFACE_CATCH_HANDLER
    auto decoder = static_cast<WenetAGDecoder*>(decoder_vp);
    auto result = decoder->RemoveGrammarFst(grammar_fst_index);
    return true;
    END_INTERFACE_CATCH_HANDLER(false)
}
