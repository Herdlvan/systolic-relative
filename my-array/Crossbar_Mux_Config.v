module Crossbar_Mux_Config #(
    parameter ARRAY_SIZE = 128
)(
    input clk,
    input reset,
    input [1:0] dataflow_mode, // 00-input stationary, 01-weight stationary, 10-output stationary
    input [8:0] config_sel,    // 387种配置选择
    output reg [1:0] ctrl_crossbar [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1],
    output reg [1:0] mux_sel [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1],
    output wire  mac_enable [0:ARRAY_SIZE-1] [0:ARRAY_SIZE-1]
);

    // 状态编码
    reg [7:0] height_size, wide_size;
    localparam S_IDLE      = 2'd0;
    localparam S_DATAFLOW  = 2'd1;
    localparam S_SIZE      = 2'd2;
    localparam S_DONE      = 2'd3;

    reg [1:0] state, next_state;
    integer i, j, k;

    // 状态转移
    always @(posedge clk or posedge reset) begin
        if (reset)
            state <= S_IDLE;
        else
            state <= next_state;
    end

    // 状态机主逻辑
    always @(*) begin
        // 默认全清
        for (j = 0; j < ARRAY_SIZE; j = j + 1)
            for (k = 0; k < ARRAY_SIZE; k = k + 1)
                ctrl_crossbar[j][k] = 2'b00;
                mac_enable[j][k]= 1'b1;
                 mux_sel[j][k] = 2'b00;


        next_state = state;
        case (state)
            S_IDLE: begin
                if (dataflow_mode != 2'b11)
                    next_state = S_DATAFLOW;
            end
            S_DATAFLOW: begin
                // 先配置数据流
                if (dataflow_mode == 2'b00) begin // input stationary
                    for (j = 0; j < ARRAY_SIZE; j = j + 1)
                        for (k = 0; k < ARRAY_SIZE; k = k + 1) begin
                            ctrl_crossbar[0][j][k] = 2'b00;
                            ctrl_crossbar[1][j][k] = 2'b00;
                            ctrl_crossbar[2][j][k] = 2'b00;
                            ctrl_crossbar[3][j][k] = 2'b00;
                            ctrl_crossbar[4][j][k] = 2'b01;
                            ctrl_crossbar[5][j][k] = 2'b01;
                            ctrl_crossbar[6][j][k] = 2'b11;
                            ctrl_crossbar[7][j][k] = 2'b01;
                            mux_sel[j][k] = 2'b01;
                            mac_enable[j][k]= 1'b1;
                        end
                end else if (dataflow_mode == 2'b01) begin // weight stationary
                    for (j = 0; j < ARRAY_SIZE; j = j + 1)
                        for (k = 0; k < ARRAY_SIZE; k = k + 1) begin
                            ctrl_crossbar[0][j][k] = 2'b00;
                            ctrl_crossbar[1][j][k] = 2'b01;
                            ctrl_crossbar[2][j][k] = 2'b00;
                            ctrl_crossbar[3][j][k] = 2'b01;
                            ctrl_crossbar[4][j][k] = 2'b00;
                            ctrl_crossbar[5][j][k] = 2'b01;
                            ctrl_crossbar[6][j][k] = 2'b11;
                            ctrl_crossbar[7][j][k] = 2'b00;
                            mux_sel[j][k] = 2'b10;
                            mac_enable[j][k]= 1'b1;
                        end
                end else if (dataflow_mode == 2'b10) begin // output stationary
                    for (j = 0; j < ARRAY_SIZE; j = j + 1)
                        for (k = 0; k < ARRAY_SIZE; k = k + 1) begin
                            ctrl_crossbar[0][j][k] = 2'b00;
                            ctrl_crossbar[1][j][k] = 2'b01;
                            ctrl_crossbar[2][j][k] = 2'b00;
                            ctrl_crossbar[3][j][k] = 2'b01;
                            ctrl_crossbar[4][j][k] = 2'b00;
                            ctrl_crossbar[5][j][k] = 2'b00;
                            ctrl_crossbar[6][j][k] = 2'b10;
                            ctrl_crossbar[7][j][k] = 2'b01;
                            mux_sel[j][k] = 2'b11;
                            mac_enable[j][k]= 1'b1;
                        end
                end
                next_state = S_SIZE;
            end
            S_SIZE: begin
                // 在当前数据流模式下，根据config_sel配置尺寸相关内容
                case (dataflow_mode)
                    2'b10: begin
                        for (j = 0; j < ARRAY_SIZE; j = j + 1)
                            for (k = 0; k < ARRAY_SIZE; k = k + 1) begin
                                if (j>smaller_size-1&&j<ARRAY_SIZE-smaller_size&&k>smaller_size-1&&k<ARRAY_SIZE-smaller_size) begin
                                    mac_enable[j][k]= 1'b0; // 禁用MAC单元
                                end
                                else if(j>=0&&j<=ARRAY_SIZE-smaller_size&&k>=ARRAY_SIZE-smaller_size&&k<=ARRAY_SIZE) begin
                                    ctrl_crossbar[0][j][k] = (j=0&&k=ARRAY_SIZE-1)?  :(j+k==ARRAY_SIZE-1)? :(j+k<ARRAY_SIZE-1)?  : // 输入选择器
                                    ctrl_crossbar[1][j][k] = (j=0&&k=ARRAY_SIZE-1)?  :(j+k==ARRAY_SIZE-1)? :(j+k<ARRAY_SIZE-1)?  : 
                                    ctrl_crossbar[2][j][k] = (j=0&&k=ARRAY_SIZE-1)?  :(j+k==ARRAY_SIZE-1)? :(j+k<ARRAY_SIZE-1)?  : 
                                    ctrl_crossbar[3][j][k] = (j=0&&k=ARRAY_SIZE-1)?  :(j+k==ARRAY_SIZE-1)? :(j+k<ARRAY_SIZE-1)?  : 
                                    ctrl_crossbar[4][j][k] = (j=0&&k=ARRAY_SIZE-1)?  :(j+k==ARRAY_SIZE-1)? :(j+k<ARRAY_SIZE-1)?  : 
                                    ctrl_crossbar[5][j][k] = (j=0&&k=ARRAY_SIZE-1)?  :(j+k==ARRAY_SIZE-1)? :(j+k<ARRAY_SIZE-1)?  : 
                                    ctrl_crossbar[6][j][k] = (j=0&&k=ARRAY_SIZE-1)?  :(j+k==ARRAY_SIZE-1)? :(j+k<ARRAY_SIZE-1)?  : 
                                    ctrl_crossbar[7][j][k] = (j=0&&k=ARRAY_SIZE-1)?  :(j+k==ARRAY_SIZE-1)? :(j+k<ARRAY_SIZE-1)?  : 
                                end
                            end
                    end
                    2'b01: begin
                        // weight stationary
                        for (i = 0; i < height_size; i = i + 1) begin
                            for (j = 0; j < wide_size; j = j + 1) begin
                                ctrl_crossbar[i][j] = 2'b10; // 设置权重选择器
                                mux_sel[i][j] = 2'b10; // 设置输入选择器
                                mac_enable[i][j] = 1'b1; // 启用MAC单元
                            end
                        end
                    end 
                    2'b00: begin
                        // input stationary
                        for (i = 0; i < height_size; i = i + 1) begin
                            for (j = 0; j < wide_size; j = j + 1) begin
                                ctrl_crossbar[i][j] = 2'b00; // 设置输入选择器
                                mux_sel[i][j] = 2'b01; // 设置输入选择器
                                mac_enable[i][j] = 1'b1; // 启用MAC单元
                            end
                        end
                    end
                endcase
                
            end
            S_DONE: begin
                // 保持配置
                next_state = S_DONE;
            end
        endcase
    end






wire [7:0] smaller_size;
    always @(*) begin
        case (config_sel)
            9'd1: begin height_size = 8'd32; wide_size = 8'd32; end
            9'd2: begin height_size = 8'd64; wide_size = 8'd64; end
            // ... 其它配置 ...
            default: begin height_size = 8'd0; wide_size = 8'd0; end
        endcase
    end
    assign smaller_size= (height_size < wide_size) ? height_size : wide_size;

endmodule