`timescale 1ns/1ps

module softmax8_tb;

    parameter DATA_WIDTH = 8;
    parameter NODES = 8; // 便于观察

    reg clk;
    reg reset;
    reg [DATA_WIDTH*NODES-1:0] inputs;
    wire [DATA_WIDTH*NODES-1:0] outputs;
    wire done;

    // 实例化 softmax8
    softmax8 #(
        .DATA_WIDTH(DATA_WIDTH),
        .NODES(NODES)
    ) dut (
        .clk(clk),
        .reset(reset),
        .inputs(inputs),
        .outputs(outputs),
        .done(done)
    );

    // 时钟生成
    initial clk = 0;
    always #5 clk = ~clk;

    integer i;

    initial begin
        $dumpfile("softmax8_tb.vcd");
        $dumpvars(0, softmax8_tb);

        // 初始化
        reset = 1;
        inputs = 0;
        #20;
        reset = 0;

        // 输入一组简单数据
        // 比如输入 1,2,3,4,5,6,7,8
        for (i = 0; i < NODES; i = i + 1)
            inputs[DATA_WIDTH*i +: DATA_WIDTH] = i + 1;

        #10;

        // 等待 done 信号
        wait(done);

        // 打印输出（Q0.8定点，除以256得到概率）
        $display("Softmax outputs (Q0.8):");
        for (i = 0; i < NODES; i = i + 1)
            $display("outputs[%0d] = %0d (prob = %f)", i, outputs[DATA_WIDTH*i +: DATA_WIDTH], outputs[DATA_WIDTH*i +: DATA_WIDTH]/256.0);

        #20;
        $finish;
    end

endmodule