module SystolicArray128x128 (
    input clk,
    input reset,
    parameter ARRAY_SIZE = 128,
    parameter DATA_WIDTH = 8,
    
    // 边界输入
    input  wire [DATA_WIDTH-1:0] data_in_bottom [0:ARRAY_SIZE-1],
    input  wire [DATA_WIDTH-1:0] data_in_left   [0:ARRAY_SIZE-1],
    input  wire [DATA_WIDTH-1:0] data_in_right  [0:ARRAY_SIZE-1],
    input  wire [DATA_WIDTH-1:0] data_in_top    [0:ARRAY_SIZE-1],
    
    // 边界输出
    output wire [DATA_WIDTH-1:0] data_out_bottom [0:ARRAY_SIZE-1],
    output wire [DATA_WIDTH-1:0] data_out_left   [0:ARRAY_SIZE-1],
    output wire [DATA_WIDTH-1:0] data_out_right  [0:ARRAY_SIZE-1],
    output wire [DATA_WIDTH-1:0] data_out_top    [0:ARRAY_SIZE-1],
    
    // 静态数据输入
    input [DATA_WIDTH-1:0] stationary_data [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1],
    
    // 控制信号 - 每个PE独立
    input [1:0] ctrl_crossbar [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1][0:7], // [i][j][0:7]
    input [1:0] mux_sel [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1],          // [i][j]
    
    // 全局控制信号
    input mac_enable,
    input accum_clear,
    input output_stationary_enable
);

// 内部PE连接信号
wire [DATA_WIDTH-1:0] pe_data_out_left   [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
wire [DATA_WIDTH-1:0] pe_data_out_right  [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
wire [DATA_WIDTH-1:0] pe_data_out_top    [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
wire [DATA_WIDTH-1:0] pe_data_out_bottom [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

wire [DATA_WIDTH-1:0] pe_data_in_left    [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
wire [DATA_WIDTH-1:0] pe_data_in_right   [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
wire [DATA_WIDTH-1:0] pe_data_in_top     [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
wire [DATA_WIDTH-1:0] pe_data_in_bottom  [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

// 生成PE阵列
generate
    for (genvar i = 0; i < ARRAY_SIZE; i = i + 1) begin : row_gen
        for (genvar j = 0; j < ARRAY_SIZE; j = j + 1) begin : col_gen
            // 连接输入
            // 左侧连接：边界输入或左侧PE的输出
            assign pe_data_in_left[i][j] = (j == 0) ? data_in_left[i] : pe_data_out_right[i][j-1];
            
            // 右侧连接：边界输入或右侧PE的输出
            assign pe_data_in_right[i][j] = (j == ARRAY_SIZE-1) ? data_in_right[i] : pe_data_out_left[i][j+1];
            
            // 顶部连接：边界输入或上方PE的输出
            assign pe_data_in_top[i][j] = (i == 0) ? data_in_top[j] : pe_data_out_bottom[i-1][j];
            
            // 底部连接：边界输入或下方PE的输出
            assign pe_data_in_bottom[i][j] = (i == ARRAY_SIZE-1) ? data_in_bottom[j] : pe_data_out_top[i+1][j];
            
            // 实例化PE单元
            PE_Unit #(
                .DATA_WIDTH(DATA_WIDTH),
                .ACCUM_WIDTH(16)  // 累加器宽度设为16位
            ) pe_inst (
                .clk(clk),
                .reset(reset),
                // 数据输入
                .data_in_left(pe_data_in_left[i][j]),
                .data_in_right(pe_data_in_right[i][j]),
                .data_in_top(pe_data_in_top[i][j]),
                .data_in_bottom(pe_data_in_bottom[i][j]),
                .Stationary_data(stationary_data[i][j]),
                // 数据输出
                .data_out_left(pe_data_out_left[i][j]),
                .data_out_right(pe_data_out_right[i][j]),
                .data_out_top(pe_data_out_top[i][j]),
                .data_out_bottom(pe_data_out_bottom[i][j]),
                // 控制信号 - 每个PE独立
                .ctrl_crossbar(ctrl_crossbar[i][j]),
                .mux_sel(mux_sel[i][j]),
                // 全局控制信号
                .mac_enable(mac_enable),
                .accum_clear(accum_clear),
                .output_stationary_enable(output_stationary_enable)
            );
        end
    end
endgenerate

// 连接边界输出
generate
    // 左侧边界输出
    for (genvar i = 0; i < ARRAY_SIZE; i = i + 1) begin : left_out
        assign data_out_left[i] = pe_data_out_left[i][0];
    end
    
    // 右侧边界输出
    for (genvar i = 0; i < ARRAY_SIZE; i = i + 1) begin : right_out
        assign data_out_right[i] = pe_data_out_right[i][ARRAY_SIZE-1];
    end
    
    // 顶部边界输出
    for (genvar j = 0; j < ARRAY_SIZE; j = j + 1) begin : top_out
        assign data_out_top[j] = pe_data_out_top[0][j];
    end
    
    // 底部边界输出
    for (genvar j = 0; j < ARRAY_SIZE; j = j + 1) begin : bottom_out
        assign data_out_bottom[j] = pe_data_out_bottom[ARRAY_SIZE-1][j];
    end
endgenerate

endmodule