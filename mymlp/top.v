module top #(
    parameter DATA_WIDTH = 8
)(
    input clk,
    input reset,
    input [9:0] M_index,
    input [9:0] N_index,
    input [9:0] K_index,
    input read_enable,
    output [DATA_WIDTH*387-1:0] softmax_out, // 改为一维向量
    output done,
    output [8:0] predicted_index
);

    // 嵌入表输出
    wire [DATA_WIDTH*8-1:0] data_out_M, data_out_N, data_out_K;
    wire [DATA_WIDTH*24-1:0] concat_out;
    wire [DATA_WIDTH*128-1:0] layer1_out;
    wire [DATA_WIDTH*128-1:0] relu_out;
    wire [DATA_WIDTH*387-1:0] layer2_out;
    wire [DATA_WIDTH*387-1:0] softmax_out_flat;
    wire softmax_done;

    // 嵌入表实例
    embedded_table_m #(.DATA_WIDTH(DATA_WIDTH)) emb_M(
        .clk(clk),
        .read_enable(read_enable),
        .index(M_index),
        .data_out(data_out_M)
    );
    embedded_table_n #(.DATA_WIDTH(DATA_WIDTH)) emb_N(
        .clk(clk),
        .read_enable(read_enable),
        .index(N_index),
        .data_out(data_out_N)
    );
    embedded_table_k #(.DATA_WIDTH(DATA_WIDTH)) emb_K(
        .clk(clk),
        .read_enable(read_enable),
        .index(K_index),
        .data_out(data_out_K)
    );

    // 拼接
    concat #(.DATA_WIDTH(DATA_WIDTH)) concat_inst(
        .clk(clk),
        .data_out_M(data_out_M),
        .data_out_N(data_out_N),
        .data_out_K(data_out_K),
        .concat_out(concat_out)
    );

    // 权重存储实例
    wire [DATA_WIDTH*128-1:0] weights1;
    wire [DATA_WIDTH*387-1:0] weights2;

    weightMemory1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .INPUT_NODES(24),
        .OUTPUT_NODES(128)
    ) weight_mem1 (
        .clk(clk),
        .address(8'd0),
        .weights(weights1)
    );

    weightMemory2 #(
        .DATA_WIDTH(DATA_WIDTH),
        .INPUT_NODES(128),
        .OUTPUT_NODES(387)
    ) weight_mem2 (
        .clk(clk),
        .address(8'd0),
        .weights(weights2)
    );

    // 第一层
    layer1 layer1_inst(
        .clk(clk),
        .reset(reset),
        .input_fc(concat_out),
        .weights(weights1),
        .output_fc(layer1_out)
    );

    // Relu
    genvar i;
    generate
        for (i = 0; i < 128; i = i + 1) begin : relu_block
            Relu relu_inst(
                .clk(clk),
                .rst_n(~reset),
                .in_data(layer1_out[DATA_WIDTH*i +: DATA_WIDTH]),
                .out_data(relu_out[DATA_WIDTH*i +: DATA_WIDTH])
            );
        end
    endgenerate

    // 第二层
    layer2 layer2_inst(
        .clk(clk),
        .reset(reset),
        .input_fc(relu_out),
        .weights(weights2),
        .output_fc(layer2_out)
    );

    // softmax
    softmax8 softmax_inst(
        .clk(clk),
        .reset(reset),
        .inputs(layer2_out),
        .outputs(softmax_out_flat),
        .done(softmax_done)
    );

    // 输出展开（直接赋值即可）
    assign softmax_out = softmax_out_flat;

    assign done = softmax_done;

    // 最大值索引输出
    reg [8:0] max_index;
    integer k;
    reg [7:0] max_value;

    always @(*) begin
        max_value = 0;
        max_index = 0;
        for (k = 0; k < 387; k = k + 1) begin
            if (softmax_out_flat[DATA_WIDTH*k +: DATA_WIDTH] > max_value) begin
                max_value = softmax_out_flat[DATA_WIDTH*k +: DATA_WIDTH];
                max_index = k[8:0];
            end
        end
    end

    assign predicted_index = max_index;

endmodule