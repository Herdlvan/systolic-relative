module softmax8 #(
    parameter DATA_WIDTH = 8,
    parameter NODES = 387
)(
    input clk,
    input reset,
    input [DATA_WIDTH*NODES-1:0] inputs,
    output reg [DATA_WIDTH*NODES-1:0] outputs, // Q0.8定点概率
    output reg done
);

    // 内部寄存器
    reg [DATA_WIDTH-1:0] input_array [0:NODES-1];
    reg [15:0] exp_array [0:NODES-1];      // 存储指数结果
    reg [23:0] exp_sum;                    // 累加所有指数
    reg [8:0] idx;
    reg [2:0] state;

    // 指数查找表
    function [15:0] exp_lut;
        input signed [7:0] x;
        reg [7:0] abs_x;
        reg [15:0] lut [0:8];
        begin
            lut[0]=16'd256;    // exp(0) * 256
            lut[1]=16'd696;    // exp(1) * 256
            lut[2]=16'd1871;   // exp(2) * 256
            lut[3]=16'd5041;   // exp(3) * 256
            lut[4]=16'd13623;  // exp(4) * 256
            lut[5]=16'd36887;  // exp(5) * 256
            lut[6]=16'd99999;  // exp(6) * 256
            lut[7]=16'd271828; // exp(7) * 256
            lut[8]=16'd738905; // exp(8) * 256
            abs_x = (x < 0) ? -x : x;
            if (x < -8)
                exp_lut = 16'd1; // exp(-大数)≈0
            else if (x > 8)
                exp_lut = lut[8]; // exp(大数)≈exp(8)
            else if (x < 0)
                exp_lut = 16'd65536 / lut[abs_x]; // 近似exp(-x)=1/exp(x)
            else
                exp_lut = lut[abs_x];
        end
    endfunction

    // 状态机
    localparam S_IDLE = 0, S_EXP = 1, S_SUM = 2, S_DIV = 3, S_DONE = 4;

    integer i;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            idx <= 0;
            state <= S_IDLE;
            exp_sum <= 0;
            done <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    for (i = 0; i < NODES; i = i + 1)
                        input_array[i] <= inputs[DATA_WIDTH*i +: DATA_WIDTH];
                    idx <= 0;
                    exp_sum <= 0;
                    done <= 0;
                    state <= S_EXP;
                end
                S_EXP: begin
                    exp_array[idx] <= exp_lut(input_array[idx]);
                    idx <= idx + 1;
                    if (idx == NODES-1) begin
                        idx <= 0;
                        state <= S_SUM;
                    end
                end
                S_SUM: begin
                    exp_sum <= 0;
                    for (i = 0; i < NODES; i = i + 1)
                        exp_sum <= exp_sum + exp_array[i];
                    idx <= 0;
                    state <= S_DIV;
                end
                S_DIV: begin
                    for (i = 0; i < NODES; i = i + 1) begin
                        // Q0.8定点概率输出
                        if (exp_sum != 0)
                            outputs[DATA_WIDTH*i +: DATA_WIDTH] <= (exp_array[i] << 8) / exp_sum;
                        else
                            outputs[DATA_WIDTH*i +: DATA_WIDTH] <= 0;
                    end
                    state <= S_DONE;
                end
                S_DONE: begin
                    done <= 1;
                end
            endcase
        end
    end

endmodule