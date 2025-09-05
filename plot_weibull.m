function plot_weibull()
% 绘制Weibull函数曲线
% Weibull函数形式: f(x; λ, k) = (k/λ)(x/λ)^(k-1)e^(-(x/λ)^k) for x ≥ 0
% 参数: 
%   k: 形状参数 (shape parameter)
%   λ: 尺度参数 (scale parameter)

    % 创建绘图窗口
    figure('Name', 'Weibull函数图像', 'Position', [100, 100, 800, 600]);
    
    % 定义x范围
    x = linspace(0, 5, 1000);
    
    % 绘制不同形状参数k下的Weibull函数
    subplot(2, 1, 1);
    lambda = 1;  % 固定尺度参数
    k_values = [0.5, 1, 1.5, 2, 3.5];
    colors = lines(length(k_values));
    
    hold on;
    for i = 1:length(k_values)
        k = k_values(i);
        y = weibull_pdf(x, lambda, k);
        plot(x, y, 'LineWidth', 2, 'Color', colors(i,:));
    end
    hold off;
    
    title('不同形状参数k下的Weibull函数 (λ = 1)', 'FontSize', 12);
    xlabel('x', 'FontSize', 11);
    ylabel('f(x; λ, k)', 'FontSize', 11);
    legend(arrayfun(@(k) sprintf('k = %.1f', k), k_values, 'UniformOutput', false), 'Location', 'northeast');
    grid on;
    
    % 绘制不同尺度参数λ下的Weibull函数
    subplot(2, 1, 2);
    k = 2;  % 固定形状参数
    lambda_values = [0.5, 1, 1.5, 2, 3];
    
    hold on;
    for i = 1:length(lambda_values)
        lambda = lambda_values(i);
        y = weibull_pdf(x, lambda, k);
        plot(x, y, 'LineWidth', 2, 'Color', colors(i,:));
    end
    hold off;
    
    title('不同尺度参数λ下的Weibull函数 (k = 2)', 'FontSize', 12);
    xlabel('x', 'FontSize', 11);
    ylabel('f(x; λ, k)', 'FontSize', 11);
    legend(arrayfun(@(lambda) sprintf('λ = %.1f', lambda), lambda_values, 'UniformOutput', false), 'Location', 'northeast');
    grid on;
    
    % 添加Weibull累积分布函数(CDF)
    figure('Name', 'Weibull累积分布函数', 'Position', [150, 150, 800, 400]);
    
    hold on;
    for i = 1:length(k_values)
        k = k_values(i);
        y = weibull_cdf(x, 1, k);
        plot(x, y, 'LineWidth', 2, 'Color', colors(i,:));
    end
    hold off;
    
    title('Weibull累积分布函数 (λ = 1)', 'FontSize', 12);
    xlabel('x', 'FontSize', 11);
    ylabel('F(x; λ, k)', 'FontSize', 11);
    legend(arrayfun(@(k) sprintf('k = %.1f', k), k_values, 'UniformOutput', false), 'Location', 'southeast');
    grid on;
    
    % 添加自定义的Weibull型函数用于奖赏机制
    figure('Name', 'Weibull型奖赏函数', 'Position', [200, 200, 800, 400]);
    
    % 创建一个类似Weibull的奖赏函数，可用于PSO算法中的奖赏机制
    x_reward = linspace(0, 1, 1000);  % 标准化到[0,1]区间
    k_reward_values = [0.5, 1, 2, 3.5, 5];
    
    hold on;
    for i = 1:length(k_reward_values)
        k = k_reward_values(i);
        % 自定义奖赏函数，基于Weibull CDF变形，使其在x=1时值为1
        y = 1 - exp(-(x_reward.^k));
        plot(x_reward, y, 'LineWidth', 2, 'Color', colors(i,:));
    end
    hold off;
    
    title('基于Weibull的奖赏函数', 'FontSize', 12);
    xlabel('标准化输入 (0-1)', 'FontSize', 11);
    ylabel('奖赏值', 'FontSize', 11);
    legend(arrayfun(@(k) sprintf('k = %.1f', k), k_reward_values, 'UniformOutput', false), 'Location', 'southeast');
    grid on;
    
    % 添加Question3中使用的特定奖赏函数: reward = 1 - exp(-(1-normalized_angle)^k)
    figure('Name', 'Question3中的奖赏函数', 'Position', [250, 250, 900, 500]);
    
    % 创建两个子图：一个以标准化角度为横轴，一个以实际角度为横轴
    subplot(1, 2, 1);
    
    % 标准化角度[0,1]，对应角度从0到pi
    x_norm = linspace(0, 1, 1000);
    k_values = [1, 1.5, 2, 2.5, 3, 4];  % 不同形状参数，包含Question3中使用的2.5
    colors = lines(length(k_values));
    
    hold on;
    highlight_idx = 4;  % k=2.5在k_values中的索引
    
    for i = 1:length(k_values)
        k = k_values(i);
        y = 1 - exp(-((1-x_norm).^k));  % Question3中使用的奖赏函数
        
        if i == highlight_idx
            % 高亮显示Question3中实际使用的k=2.5曲线
            plot(x_norm, y, 'LineWidth', 3, 'Color', colors(i,:));
        else
            plot(x_norm, y, 'LineWidth', 1.5, 'Color', colors(i,:));
        end
    end
    hold off;
    
    title('Question3中使用的奖赏函数 (标准化角度)', 'FontSize', 12);
    xlabel('标准化角度 (0=方向一致，1=方向相反)', 'FontSize', 11);
    ylabel('奖赏值', 'FontSize', 11);
    legend(arrayfun(@(k) sprintf('k = %.1f%s', k, iff(k==2.5,' (使用值)','')), k_values, 'UniformOutput', false), 'Location', 'northeast');
    grid on;
    
    % 子图2：使用实际角度为横轴（0到π）
    subplot(1, 2, 2);
    
    % 实际角度[0,pi]
    angles = linspace(0, pi, 1000);
    
    hold on;
    for i = 1:length(k_values)
        k = k_values(i);
        normalized_angles = angles / pi;
        y = 1 - exp(-((1-normalized_angles).^k));  % Question3中使用的奖赏函数
        
        if i == highlight_idx
            % 高亮显示Question3中实际使用的k=2.5曲线
            plot(angles, y, 'LineWidth', 3, 'Color', colors(i,:));
        else
            plot(angles, y, 'LineWidth', 1.5, 'Color', colors(i,:));
        end
    end
    hold off;
    
    title('Question3中使用的奖赏函数 (实际角度)', 'FontSize', 12);
    xlabel('角度 (弧度)', 'FontSize', 11);
    ylabel('奖赏值', 'FontSize', 11);
    
    % 添加角度刻度标签
    ax = gca;
    ax.XTick = [0, pi/6, pi/4, pi/3, pi/2, 2*pi/3, pi];
    ax.XTickLabel = {'0', '\pi/6', '\pi/4', '\pi/3', '\pi/2', '2\pi/3', '\pi'};
    
    legend(arrayfun(@(k) sprintf('k = %.1f%s', k, iff(k==2.5,' (使用值)','')), k_values, 'UniformOutput', false), 'Location', 'northeast');
    grid on;
    
    % 添加修改后的Question3中使用的基于Weibull生存函数的奖赏函数
    figure('Name', 'Question3中的Weibull生存函数奖赏函数', 'Position', [300, 300, 900, 500]);
    
    % 创建两个子图：一个以标准化角度为横轴，一个以实际角度为横轴
    subplot(1, 2, 1);
    
    % 标准化角度[0,1]，对应角度从0到pi
    x_norm = linspace(0, 1, 1000);
    lambda_values = [0.3, 0.5, 0.8, 1.0, 1.5];
    k_fixed = 2.5;  % 固定形状参数
    colors = lines(length(lambda_values));
    
    hold on;
    highlight_idx = 2;  % lambda=0.5在lambda_values中的索引
    
    for i = 1:length(lambda_values)
        lambda = lambda_values(i);
        y = exp(-((x_norm/lambda).^k_fixed));  % 修改后的奖赏函数（Weibull生存函数）
        
        if i == highlight_idx
            % 高亮显示使用的lambda=0.5曲线
            plot(x_norm, y, 'LineWidth', 3, 'Color', colors(i,:));
        else
            plot(x_norm, y, 'LineWidth', 1.5, 'Color', colors(i,:));
        end
    end
    hold off;
    
    title('Weibull生存函数奖赏 (k=2.5, 不同λ值)', 'FontSize', 12);
    xlabel('标准化角度 (0=方向一致，1=方向相反)', 'FontSize', 11);
    ylabel('奖赏值', 'FontSize', 11);
    legend(arrayfun(@(lambda) sprintf('λ = %.1f%s', lambda, iff(lambda==0.5,' (使用值)','')), lambda_values, 'UniformOutput', false), 'Location', 'northeast');
    grid on;
    
    % 子图2：固定lambda，不同k值
    subplot(1, 2, 2);
    
    % 实际角度[0,pi]
    angles = linspace(0, pi, 1000);
    k_values = [1, 1.5, 2, 2.5, 3, 4];
    lambda_fixed = 0.5;  % 固定尺度参数
    
    hold on;
    highlight_idx = 4;  % k=2.5在k_values中的索引
    
    for i = 1:length(k_values)
        k = k_values(i);
        normalized_angles = angles / pi;
        y = exp(-((normalized_angles/lambda_fixed).^k));  % 修改后的奖赏函数（Weibull生存函数）
        
        if i == highlight_idx
            % 高亮显示使用的k=2.5曲线
            plot(angles, y, 'LineWidth', 3, 'Color', colors(i,:));
        else
            plot(angles, y, 'LineWidth', 1.5, 'Color', colors(i,:));
        end
    end
    hold off;
    
    title('Weibull生存函数奖赏 (λ=0.5, 不同k值)', 'FontSize', 12);
    xlabel('角度 (弧度)', 'FontSize', 11);
    ylabel('奖赏值', 'FontSize', 11);
    
    % 添加角度刻度标签
    ax = gca;
    ax.XTick = [0, pi/6, pi/4, pi/3, pi/2, 2*pi/3, pi];
    ax.XTickLabel = {'0', '\pi/6', '\pi/4', '\pi/3', '\pi/2', '2\pi/3', '\pi'};
    
    legend(arrayfun(@(k) sprintf('k = %.1f%s', k, iff(k==2.5,' (使用值)','')), k_values, 'UniformOutput', false), 'Location', 'northeast');
    grid on;
end

% Weibull概率密度函数(PDF)
function y = weibull_pdf(x, lambda, k)
    % 确保x非负
    x = max(x, 0);
    
    % 计算Weibull PDF: f(x; λ, k) = (k/λ)(x/λ)^(k-1)e^(-(x/λ)^k)
    y = (k/lambda) .* (x/lambda).^(k-1) .* exp(-((x/lambda).^k));
    
    % 处理x=0的特殊情况
    if k < 1
        y(x == 0) = Inf;  % 当k<1时，x=0处函数值为无穷大
    elseif k == 1
        y(x == 0) = 1/lambda;  % 当k=1时，f(0)=1/λ
    else
        y(x == 0) = 0;  % 当k>1时，f(0)=0
    end
end

% Weibull累积分布函数(CDF)
function y = weibull_cdf(x, lambda, k)
    % 确保x非负
    x = max(x, 0);
    
    % 计算Weibull CDF: F(x; λ, k) = 1 - e^(-(x/λ)^k)
    y = 1 - exp(-((x/lambda).^k));
end 

% 条件函数，用于标记高亮文本
function result = iff(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end 