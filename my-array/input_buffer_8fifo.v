module input_buffer_8fifo #(
    parameter DATA_WIDTH = 64,
    parameter FIFO_DEPTH = 4
)(
    input clk,
    input rst_n,
    input [15:0] wr_en,                // 16路写使能
    input [15:0] rd_en,                // 16路读使能
    input [DATA_WIDTH*16-1:0] din,     // 16路输入，每路64位
    output [8*128-1:0] dout,           // 128个8位输出
    output [15:0] empty,
    output [15:0] full
);

    genvar i;
    wire [DATA_WIDTH-1:0] fifo_out [0:15];

    // 16个FIFO实例
    generate
        for (i = 0; i < 16; i = i + 1) begin : fifo_gen
            sync_fifo #(
                .DATA_WIDTH(DATA_WIDTH),
                .DEPTH(FIFO_DEPTH)
            ) fifo_inst (
                .clk(clk),
                .rst_n(rst_n),
                .wr_en(wr_en[i]),
                .rd_en(rd_en[i]),
                .din(din[DATA_WIDTH*i +: DATA_WIDTH]),
                .dout(fifo_out[i]),
                .empty(empty[i]),
                .full(full[i])
            );
        end
    endgenerate

    // 拆分每个FIFO的64位输出为8个8位，拼接成128个8位输出
    generate
        for (i = 0; i < 16; i = i + 1) begin : dout_gen
            assign dout[i*64 +: 64] = fifo_out[i];
        end
    endgenerate

endmodule