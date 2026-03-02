from transformers.configuration_utils import PretrainedConfig

class SMoEConfig(PretrainedConfig):
    model_type = "smoe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        use_cache=True,
        attention_bias=False,
        lm_head_bias=False,
        initializer_range=0.02,
        vocab_size=262272,
        hidden_size=2048,
        ffn_hidden_size_list=[0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096, 0, 4096],
        num_hidden_layers=120,
        num_attention_heads=16,
        num_attention_heads_list= [16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0,  16, 0 ],
        activation_func='swiglu',
        max_position_embeddings=4096,
        norm_epsilon=1e-05,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rotary_base=10000,
        use_rope_scaling=False,
        attention_dropout=0.0,
        moe_router_topk=1,
        smoe_layers=['a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16, 'a', 16],
        normalization='RMSNorm',
        smoe_mlp_expansion=[0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256, 0, 256],
        smoe_use_mod=True,
        smoe_mod_per=0.0,
        smoe_high_prec=True,
        smoe_inhibitory=False,
        smoe_p_switch=0.0,
        smoe_use_eda=True,
        smoe_parallel_scan_thres=128,
        add_bias_linear=False,
        gated_linear_unit=True,
        scale_residual_merge=True,
        fused_add_norm=False,
        residual_in_fp32=False,
        apply_rope_fusion=True,
        ar_threshold=1,
        bias_activation_fusion=True,
        activation_func_fp8_input_store=False,
        sliding_window=None,
        clamp_temp=False,
        num_key_value_heads=2,
        mamba_chunk_size=16,
        chunk_size=16,
        ignore_mod_in_smoe_block=False, # Make sure you understand to enable this only when the model is really not using MoD even though MoD is technically included
        **kwargs,
    ):
        self.cca=True
        self.num_attention_heads_list=num_attention_heads_list
        self.cca_num_q_heads= [8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0, 8,0]
        self.num_query_groups_list=[ 2, 0, 2, 0, 2, 0 ,2, 0, 2, 0, 2, 0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0, 2,0]
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.lm_head_bias = lm_head_bias
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size_list = ffn_hidden_size_list
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        assert self.hidden_size % self.num_attention_heads == 0
        self.kv_channels = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.activation_func = activation_func
        self.max_position_embeddings = max_position_embeddings
        self.norm_epsilon = norm_epsilon
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rotary_base = rotary_base
        self.use_rope_scaling = use_rope_scaling
        self.attention_dropout = attention_dropout
        self.moe_router_topk = moe_router_topk
        self.smoe_layers = smoe_layers
        self.use_lora_att = False
        self.lora_rank = 0
        self.normalization = normalization
        self.smoe_mlp_expansion = smoe_mlp_expansion
        self.smoe_use_mod = smoe_use_mod
        self.smoe_mod_per = smoe_mod_per
        self.smoe_high_prec = smoe_high_prec
        self.smoe_inhibitory = smoe_inhibitory
        self.smoe_p_switch = smoe_p_switch
        self.smoe_use_eda = smoe_use_eda
        self.smoe_parallel_scan_thres = smoe_parallel_scan_thres
        self.add_bias_linear = add_bias_linear
        self.gated_linear_unit = gated_linear_unit
        self.scale_residual_merge = scale_residual_merge
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.apply_rope_fusion = apply_rope_fusion
        self.ar_threshold = ar_threshold
        self.bias_activation_fusion = bias_activation_fusion
        self.activation_func_fp8_input_store = activation_func_fp8_input_store
        self.sliding_window = sliding_window
        self.init_from_megatron_checkpoint = False
        self.init_from_megatron_checkpoint_path = '/workspace/praneeth_temp/iter_0010000/mp_rank_00/model_optim_rng.pt'
        self.rope_scaling = None
        # self.rope_pct = 0.5
        # self.rope_theta = 10000.0
        self.num_key_value_heads = num_key_value_heads
        self.clamp_temp = clamp_temp
        self.ignore_mod_in_smoe_block = ignore_mod_in_smoe_block
        self.mamba_chunk_size = mamba_chunk_size
        self.chunk_size = chunk_size

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            **kwargs,
        )