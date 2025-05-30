LIB_OBJS_GEN = $(patsubst %, $(OBJDIR)/genobj/%.o, all_gather.cu all_reduce_minmax_bf16.cu all_reduce_minmax_f16.cu all_reduce_minmax_f32.cu all_reduce_minmax_f64.cu all_reduce_minmax_f8e4m3.cu all_reduce_minmax_f8e5m2.cu all_reduce_minmax_i32.cu all_reduce_minmax_i64.cu all_reduce_minmax_u32.cu all_reduce_minmax_u64.cu all_reduce_minmax_u8.cu all_reduce_premulsum_bf16.cu all_reduce_premulsum_f16.cu all_reduce_premulsum_f32.cu all_reduce_premulsum_f64.cu all_reduce_premulsum_f8e4m3.cu all_reduce_premulsum_f8e5m2.cu all_reduce_premulsum_u32.cu all_reduce_premulsum_u64.cu all_reduce_premulsum_u8.cu all_reduce_prod_bf16.cu all_reduce_prod_f16.cu all_reduce_prod_f32.cu all_reduce_prod_f64.cu all_reduce_prod_f8e4m3.cu all_reduce_prod_f8e5m2.cu all_reduce_prod_u32.cu all_reduce_prod_u64.cu all_reduce_prod_u8.cu all_reduce_sum_bf16.cu all_reduce_sum_f16.cu all_reduce_sum_f32.cu all_reduce_sum_f64.cu all_reduce_sum_f8e4m3.cu all_reduce_sum_f8e5m2.cu all_reduce_sum_u32.cu all_reduce_sum_u64.cu all_reduce_sum_u8.cu all_reduce_sumpostdiv_u32.cu all_reduce_sumpostdiv_u64.cu all_reduce_sumpostdiv_u8.cu broadcast.cu reduce_minmax_bf16.cu reduce_minmax_f16.cu reduce_minmax_f32.cu reduce_minmax_f64.cu reduce_minmax_f8e4m3.cu reduce_minmax_f8e5m2.cu reduce_minmax_u32.cu reduce_minmax_u64.cu reduce_minmax_u8.cu reduce_premulsum_bf16.cu reduce_premulsum_f16.cu reduce_premulsum_f32.cu reduce_premulsum_f64.cu reduce_premulsum_f8e4m3.cu reduce_premulsum_f8e5m2.cu reduce_premulsum_u32.cu reduce_premulsum_u64.cu reduce_premulsum_u8.cu reduce_prod_bf16.cu reduce_prod_f16.cu reduce_prod_f32.cu reduce_prod_f64.cu reduce_prod_f8e4m3.cu reduce_prod_f8e5m2.cu reduce_prod_u32.cu reduce_prod_u64.cu reduce_prod_u8.cu reduce_scatter_minmax_bf16.cu reduce_scatter_minmax_f16.cu reduce_scatter_minmax_f32.cu reduce_scatter_minmax_f64.cu reduce_scatter_minmax_f8e4m3.cu reduce_scatter_minmax_f8e5m2.cu reduce_scatter_minmax_i32.cu reduce_scatter_minmax_i64.cu reduce_scatter_minmax_u32.cu reduce_scatter_minmax_u64.cu reduce_scatter_minmax_u8.cu reduce_scatter_premulsum_bf16.cu reduce_scatter_premulsum_f16.cu reduce_scatter_premulsum_f32.cu reduce_scatter_premulsum_f64.cu reduce_scatter_premulsum_f8e4m3.cu reduce_scatter_premulsum_f8e5m2.cu reduce_scatter_premulsum_u32.cu reduce_scatter_premulsum_u64.cu reduce_scatter_premulsum_u8.cu reduce_scatter_prod_bf16.cu reduce_scatter_prod_f16.cu reduce_scatter_prod_f32.cu reduce_scatter_prod_f64.cu reduce_scatter_prod_f8e4m3.cu reduce_scatter_prod_f8e5m2.cu reduce_scatter_prod_u32.cu reduce_scatter_prod_u64.cu reduce_scatter_prod_u8.cu reduce_scatter_sum_bf16.cu reduce_scatter_sum_f16.cu reduce_scatter_sum_f32.cu reduce_scatter_sum_f64.cu reduce_scatter_sum_f8e4m3.cu reduce_scatter_sum_f8e5m2.cu reduce_scatter_sum_u32.cu reduce_scatter_sum_u64.cu reduce_scatter_sum_u8.cu reduce_scatter_sumpostdiv_u32.cu reduce_scatter_sumpostdiv_u64.cu reduce_scatter_sumpostdiv_u8.cu reduce_sum_bf16.cu reduce_sum_f16.cu reduce_sum_f32.cu reduce_sum_f64.cu reduce_sum_f8e4m3.cu reduce_sum_f8e5m2.cu reduce_sum_u32.cu reduce_sum_u64.cu reduce_sum_u8.cu reduce_sumpostdiv_u32.cu reduce_sumpostdiv_u64.cu reduce_sumpostdiv_u8.cu sendrecv.cu host_table.cc device_table.cu)

$(OBJDIR)/genobj/all_gather.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_gather.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_gather.cu)

$(OBJDIR)/genobj/all_reduce_minmax_bf16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_minmax_bf16.cu)

$(OBJDIR)/genobj/all_reduce_minmax_f16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_minmax_f16.cu)

$(OBJDIR)/genobj/all_reduce_minmax_f32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_minmax_f32.cu)

$(OBJDIR)/genobj/all_reduce_minmax_f64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_minmax_f64.cu)

$(OBJDIR)/genobj/all_reduce_minmax_f8e4m3.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_minmax_f8e4m3.cu)

$(OBJDIR)/genobj/all_reduce_minmax_f8e5m2.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_minmax_f8e5m2.cu)

$(OBJDIR)/genobj/all_reduce_minmax_i32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_minmax_i32.cu)

$(OBJDIR)/genobj/all_reduce_minmax_i64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_minmax_i64.cu)

$(OBJDIR)/genobj/all_reduce_minmax_u32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_minmax_u32.cu)

$(OBJDIR)/genobj/all_reduce_minmax_u64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_minmax_u64.cu)

$(OBJDIR)/genobj/all_reduce_minmax_u8.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_minmax_u8.cu)

$(OBJDIR)/genobj/all_reduce_premulsum_bf16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_premulsum_bf16.cu)

$(OBJDIR)/genobj/all_reduce_premulsum_f16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_premulsum_f16.cu)

$(OBJDIR)/genobj/all_reduce_premulsum_f32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_premulsum_f32.cu)

$(OBJDIR)/genobj/all_reduce_premulsum_f64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_premulsum_f64.cu)

$(OBJDIR)/genobj/all_reduce_premulsum_f8e4m3.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_premulsum_f8e4m3.cu)

$(OBJDIR)/genobj/all_reduce_premulsum_f8e5m2.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_premulsum_f8e5m2.cu)

$(OBJDIR)/genobj/all_reduce_premulsum_u32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_premulsum_u32.cu)

$(OBJDIR)/genobj/all_reduce_premulsum_u64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_premulsum_u64.cu)

$(OBJDIR)/genobj/all_reduce_premulsum_u8.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_premulsum_u8.cu)

$(OBJDIR)/genobj/all_reduce_prod_bf16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_prod_bf16.cu)

$(OBJDIR)/genobj/all_reduce_prod_f16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_prod_f16.cu)

$(OBJDIR)/genobj/all_reduce_prod_f32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_prod_f32.cu)

$(OBJDIR)/genobj/all_reduce_prod_f64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_prod_f64.cu)

$(OBJDIR)/genobj/all_reduce_prod_f8e4m3.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_prod_f8e4m3.cu)

$(OBJDIR)/genobj/all_reduce_prod_f8e5m2.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_prod_f8e5m2.cu)

$(OBJDIR)/genobj/all_reduce_prod_u32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_prod_u32.cu)

$(OBJDIR)/genobj/all_reduce_prod_u64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_prod_u64.cu)

$(OBJDIR)/genobj/all_reduce_prod_u8.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_prod_u8.cu)

$(OBJDIR)/genobj/all_reduce_sum_bf16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_sum_bf16.cu)

$(OBJDIR)/genobj/all_reduce_sum_f16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_sum_f16.cu)

$(OBJDIR)/genobj/all_reduce_sum_f32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_sum_f32.cu)

$(OBJDIR)/genobj/all_reduce_sum_f64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_sum_f64.cu)

$(OBJDIR)/genobj/all_reduce_sum_f8e4m3.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_sum_f8e4m3.cu)

$(OBJDIR)/genobj/all_reduce_sum_f8e5m2.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_sum_f8e5m2.cu)

$(OBJDIR)/genobj/all_reduce_sum_u32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_sum_u32.cu)

$(OBJDIR)/genobj/all_reduce_sum_u64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_sum_u64.cu)

$(OBJDIR)/genobj/all_reduce_sum_u8.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_sum_u8.cu)

$(OBJDIR)/genobj/all_reduce_sumpostdiv_u32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_sumpostdiv_u32.cu)

$(OBJDIR)/genobj/all_reduce_sumpostdiv_u64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_sumpostdiv_u64.cu)

$(OBJDIR)/genobj/all_reduce_sumpostdiv_u8.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/all_reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/all_reduce_sumpostdiv_u8.cu)

$(OBJDIR)/genobj/broadcast.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/broadcast.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/broadcast.cu)

$(OBJDIR)/genobj/reduce_minmax_bf16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_minmax_bf16.cu)

$(OBJDIR)/genobj/reduce_minmax_f16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_minmax_f16.cu)

$(OBJDIR)/genobj/reduce_minmax_f32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_minmax_f32.cu)

$(OBJDIR)/genobj/reduce_minmax_f64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_minmax_f64.cu)

$(OBJDIR)/genobj/reduce_minmax_f8e4m3.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_minmax_f8e4m3.cu)

$(OBJDIR)/genobj/reduce_minmax_f8e5m2.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_minmax_f8e5m2.cu)

$(OBJDIR)/genobj/reduce_minmax_u32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_minmax_u32.cu)

$(OBJDIR)/genobj/reduce_minmax_u64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_minmax_u64.cu)

$(OBJDIR)/genobj/reduce_minmax_u8.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_minmax_u8.cu)

$(OBJDIR)/genobj/reduce_premulsum_bf16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_premulsum_bf16.cu)

$(OBJDIR)/genobj/reduce_premulsum_f16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_premulsum_f16.cu)

$(OBJDIR)/genobj/reduce_premulsum_f32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_premulsum_f32.cu)

$(OBJDIR)/genobj/reduce_premulsum_f64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_premulsum_f64.cu)

$(OBJDIR)/genobj/reduce_premulsum_f8e4m3.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_premulsum_f8e4m3.cu)

$(OBJDIR)/genobj/reduce_premulsum_f8e5m2.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_premulsum_f8e5m2.cu)

$(OBJDIR)/genobj/reduce_premulsum_u32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_premulsum_u32.cu)

$(OBJDIR)/genobj/reduce_premulsum_u64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_premulsum_u64.cu)

$(OBJDIR)/genobj/reduce_premulsum_u8.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_premulsum_u8.cu)

$(OBJDIR)/genobj/reduce_prod_bf16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_prod_bf16.cu)

$(OBJDIR)/genobj/reduce_prod_f16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_prod_f16.cu)

$(OBJDIR)/genobj/reduce_prod_f32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_prod_f32.cu)

$(OBJDIR)/genobj/reduce_prod_f64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_prod_f64.cu)

$(OBJDIR)/genobj/reduce_prod_f8e4m3.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_prod_f8e4m3.cu)

$(OBJDIR)/genobj/reduce_prod_f8e5m2.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_prod_f8e5m2.cu)

$(OBJDIR)/genobj/reduce_prod_u32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_prod_u32.cu)

$(OBJDIR)/genobj/reduce_prod_u64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_prod_u64.cu)

$(OBJDIR)/genobj/reduce_prod_u8.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_prod_u8.cu)

$(OBJDIR)/genobj/reduce_scatter_minmax_bf16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_minmax_bf16.cu)

$(OBJDIR)/genobj/reduce_scatter_minmax_f16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_minmax_f16.cu)

$(OBJDIR)/genobj/reduce_scatter_minmax_f32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_minmax_f32.cu)

$(OBJDIR)/genobj/reduce_scatter_minmax_f64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_minmax_f64.cu)

$(OBJDIR)/genobj/reduce_scatter_minmax_f8e4m3.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_minmax_f8e4m3.cu)

$(OBJDIR)/genobj/reduce_scatter_minmax_f8e5m2.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_minmax_f8e5m2.cu)

$(OBJDIR)/genobj/reduce_scatter_minmax_i32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_minmax_i32.cu)

$(OBJDIR)/genobj/reduce_scatter_minmax_i64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_minmax_i64.cu)

$(OBJDIR)/genobj/reduce_scatter_minmax_u32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_minmax_u32.cu)

$(OBJDIR)/genobj/reduce_scatter_minmax_u64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_minmax_u64.cu)

$(OBJDIR)/genobj/reduce_scatter_minmax_u8.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_minmax_u8.cu)

$(OBJDIR)/genobj/reduce_scatter_premulsum_bf16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_premulsum_bf16.cu)

$(OBJDIR)/genobj/reduce_scatter_premulsum_f16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_premulsum_f16.cu)

$(OBJDIR)/genobj/reduce_scatter_premulsum_f32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_premulsum_f32.cu)

$(OBJDIR)/genobj/reduce_scatter_premulsum_f64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_premulsum_f64.cu)

$(OBJDIR)/genobj/reduce_scatter_premulsum_f8e4m3.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_premulsum_f8e4m3.cu)

$(OBJDIR)/genobj/reduce_scatter_premulsum_f8e5m2.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_premulsum_f8e5m2.cu)

$(OBJDIR)/genobj/reduce_scatter_premulsum_u32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_premulsum_u32.cu)

$(OBJDIR)/genobj/reduce_scatter_premulsum_u64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_premulsum_u64.cu)

$(OBJDIR)/genobj/reduce_scatter_premulsum_u8.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_premulsum_u8.cu)

$(OBJDIR)/genobj/reduce_scatter_prod_bf16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_prod_bf16.cu)

$(OBJDIR)/genobj/reduce_scatter_prod_f16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_prod_f16.cu)

$(OBJDIR)/genobj/reduce_scatter_prod_f32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_prod_f32.cu)

$(OBJDIR)/genobj/reduce_scatter_prod_f64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_prod_f64.cu)

$(OBJDIR)/genobj/reduce_scatter_prod_f8e4m3.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_prod_f8e4m3.cu)

$(OBJDIR)/genobj/reduce_scatter_prod_f8e5m2.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_prod_f8e5m2.cu)

$(OBJDIR)/genobj/reduce_scatter_prod_u32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_prod_u32.cu)

$(OBJDIR)/genobj/reduce_scatter_prod_u64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_prod_u64.cu)

$(OBJDIR)/genobj/reduce_scatter_prod_u8.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_prod_u8.cu)

$(OBJDIR)/genobj/reduce_scatter_sum_bf16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_sum_bf16.cu)

$(OBJDIR)/genobj/reduce_scatter_sum_f16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_sum_f16.cu)

$(OBJDIR)/genobj/reduce_scatter_sum_f32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_sum_f32.cu)

$(OBJDIR)/genobj/reduce_scatter_sum_f64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_sum_f64.cu)

$(OBJDIR)/genobj/reduce_scatter_sum_f8e4m3.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_sum_f8e4m3.cu)

$(OBJDIR)/genobj/reduce_scatter_sum_f8e5m2.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_sum_f8e5m2.cu)

$(OBJDIR)/genobj/reduce_scatter_sum_u32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_sum_u32.cu)

$(OBJDIR)/genobj/reduce_scatter_sum_u64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_sum_u64.cu)

$(OBJDIR)/genobj/reduce_scatter_sum_u8.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_sum_u8.cu)

$(OBJDIR)/genobj/reduce_scatter_sumpostdiv_u32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_sumpostdiv_u32.cu)

$(OBJDIR)/genobj/reduce_scatter_sumpostdiv_u64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_sumpostdiv_u64.cu)

$(OBJDIR)/genobj/reduce_scatter_sumpostdiv_u8.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce_scatter.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_scatter_sumpostdiv_u8.cu)

$(OBJDIR)/genobj/reduce_sum_bf16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_sum_bf16.cu)

$(OBJDIR)/genobj/reduce_sum_f16.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_sum_f16.cu)

$(OBJDIR)/genobj/reduce_sum_f32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_sum_f32.cu)

$(OBJDIR)/genobj/reduce_sum_f64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_sum_f64.cu)

$(OBJDIR)/genobj/reduce_sum_f8e4m3.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_sum_f8e4m3.cu)

$(OBJDIR)/genobj/reduce_sum_f8e5m2.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_sum_f8e5m2.cu)

$(OBJDIR)/genobj/reduce_sum_u32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_sum_u32.cu)

$(OBJDIR)/genobj/reduce_sum_u64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_sum_u64.cu)

$(OBJDIR)/genobj/reduce_sum_u8.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_sum_u8.cu)

$(OBJDIR)/genobj/reduce_sumpostdiv_u32.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_sumpostdiv_u32.cu)

$(OBJDIR)/genobj/reduce_sumpostdiv_u64.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_sumpostdiv_u64.cu)

$(OBJDIR)/genobj/reduce_sumpostdiv_u8.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/reduce.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/reduce_sumpostdiv_u8.cu)

$(OBJDIR)/genobj/sendrecv.cu.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/sendrecv.cu.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/sendrecv.cu)

