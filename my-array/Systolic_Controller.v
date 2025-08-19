module Systolic_Controller #(
    parameter ARRAY_SIZE = 128
)(
    input clk,
    input reset,
    input start,
    input [1:0] dataflow_mode, // 00-input, 01-weight, 10-output stationary
    output reg [1:0] ctrl_crossbar [0:7][0:ARRAY_SIZE-1][0:ARRAY_SIZE-1],
    output reg [1:0] mux_sel [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1],
    output reg mac_enable [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1],
    output reg accum_clear [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1],
    output reg done
);

    typedef enum logic [1:0] {
        IDLE,
        LOAD_STATIONARY,
        COMPUTE_AND_DRAIN,
        DONE
    } state_t;

    state_t state, next_state;
    integer i, j, t;

    // 状态转移
    always @(posedge clk or posedge reset) begin
        if (reset)
            state <= IDLE;
        else
            state <= next_state;
    end

    // 状态机主逻辑
    always @(*) begin
        // 默认值
        done = 0;
        for (i = 0; i < ARRAY_SIZE; i = i + 1)
            for (j = 0; j < ARRAY_SIZE; j = j + 1) begin
                ctrl_crossbar[0][i][j] = 2'b00; // 具体配置按你的crossbar定义
                mux_sel[i][j] = 2'b00;
                mac_enable[i][j] = 0;
                accum_clear[i][j] = 0;
            end

        case (state)
            IDLE: begin
                if (start)
                    next_state = (dataflow_mode == 2'b10) ? COMPUTE_AND_DRAIN : LOAD_STATIONARY;
                else
                    next_state = IDLE;
            end
            LOAD_STATIONARY: begin
                // 配置crossbar和mux，让input或weight进入PE内部寄存器
                // 拉高accum_clear，拉低mac_enable
                // 只在input/weight stationary模式下有效
                // ...配置...
                next_state = COMPUTE_AND_DRAIN;
            end
            COMPUTE_AND_DRAIN: begin
                // 配置crossbar和mux，让流动数据进入MAC
                // mac_enable拉高，accum_clear拉低
                // 输出随时可采集
                // ...配置...
                // 判断所有数据流动完毕后进入DONE
                if (/* 计算和输出全部完成条件 */)
                    next_state = DONE;
                else
                    next_state = COMPUTE_AND_DRAIN;
            end
            DONE: begin
                done = 1;
                next_state = IDLE;
            end
        endcase
    end

endmodule