
module mux_top_compact (
    // 选择信号数组
    input wire [6:0] sel [0:63], 
    
    // 输入数据数组
    input wire [7:0] in [0:63][0:64], // 第一维是MUX索引，第二维是输入
    
    // 输出数组
    output wire [7:0] out [0:63]
);

    genvar i;
    generate
        for (i = 0; i < 64; i = i + 1) begin : mux_gen
            localparam INPUT_NUM = 65 - i;
            
            param_mux #(
                .DATA_WIDTH(8),
                .INPUT_NUM(INPUT_NUM)
            ) u_mux (
                .sel(sel[i][$clog2(INPUT_NUM)-1:0]),
                .in(in[i][0:INPUT_NUM-1]),
                .out(out[i])
            );
        end
    endgenerate

endmodule

module param_mux #(
    parameter DATA_WIDTH = 1,     // 数据线宽度(默认为1位)
    parameter INPUT_NUM  = 2      // 输入数量(默认为2输入)
) (
    input wire [$clog2(INPUT_NUM)-1:0] sel,  // 选择信号，自动计算所需位数
    input wire [DATA_WIDTH-1:0] in [0:INPUT_NUM-1], // 输入端口数组
    output reg [DATA_WIDTH-1:0] out           // 输出端口
);

    always @(*) begin
        out = in[sel];  // 根据选择信号选择对应输入
    end

endmodule