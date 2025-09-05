% 粒子群优化（PSO）：最大化第一问遮挡总时长（速度=大小s+方位角theta）
% 目标函数：q1_occlusion_time(vx_FY1, vy_FY1, t_throw, t_explode)

clear; clc;
% 获取当前随机数生成器状态
current_state = rng;

% 提取并输出随机种子
current_seed = current_state.Seed;
fprintf('当前随机种子为：%d\n', current_seed);
%% 约束与场景参数
% 速度大小与方位角
SPEED_MIN = 70; SPEED_MAX = 140; % s ∈ [70,140]
THETA_MIN = -pi; THETA_MAX = pi; % theta ∈ [-pi, pi]

% von Mises 方向生成参数（指向 x 轴负半轴）
VM_mu = pi;        % 均值方向：π（指向 -x 方向）
VM_kappa = 10;      % 浓度参数

% 计算导弹到达目标的时间（t_throw + t_explode 必须小于该时间）
T_MAX = 20;
T_SUM_MAX = 20 * (1 - 1e-6);            % 略微留余量，保证严格小于

% 目标函数仿真选项（传给 q1_occlusion_time → computeOcclusionSimple）
q1_opts = struct('tSimEnd', 20, 'g', 9.8);

%% 粒子群参数与搜索边界
% 变量：[s theta t_throw t_explode]
nVar = 4;
VarMin = [SPEED_MIN, THETA_MIN, 0.0, 0.0];
VarMax = [SPEED_MAX, THETA_MAX, T_MAX, T_MAX];

nPop = 50;              % 群体规模
MaxIt = 120;            % 最大迭代次数
w = 0.72;               % 惯性权重
wDamp = 0.99;           % 惯性权重衰减
c1 = 1.8;               % 个体学习因子
c2 = 1.8;               % 群体学习因子

% 速度上限（按搜索区间的一定比例设定）
VelMax = 0.25 * (VarMax - VarMin);
VelMin = -VelMax;

%% 目标函数（最小化：-遮挡时长；时间强约束在外部“修复”）
obj = @(x) objective_s_theta(x, q1_opts);

%% 粒子结构体
empty_particle.Position = [];
empty_particle.Velocity = [];
empty_particle.Cost = [];
empty_particle.Duration = [];
empty_particle.Best.Position = [];
empty_particle.Best.Cost = [];
empty_particle.Best.Duration = [];

% 初始化群体（theta ~ von Mises(mu=π, kappa=8)）
particle = repmat(empty_particle, nPop, 1);
GlobalBest.Cost = inf;
GlobalBest.Duration = -inf;
GlobalBest.Position = [];

for i = 1:nPop
    particle(i).Position = VarMin + rand(1, nVar) .* (VarMax - VarMin);
    % 用 von Mises 分布采样方位角；速度大小均匀
    particle(i).Position(1) = SPEED_MIN + rand*(SPEED_MAX - SPEED_MIN);         % s
    particle(i).Position(2) = randVonMises(VM_mu, VM_kappa, 1);                 % theta ∈ [-pi,pi]
    % 强可行修复：保证 t0+t1 <= T_SUM_MAX
    t0i = particle(i).Position(3); t1i = particle(i).Position(4);
    tsum = t0i + t1i;
    if tsum > T_SUM_MAX
        sf = T_SUM_MAX / tsum; particle(i).Position(3:4) = particle(i).Position(3:4) * sf;
    end
    particle(i).Velocity = zeros(1, nVar);

    [J, dur] = obj(particle(i).Position);
    particle(i).Cost = J;
    particle(i).Duration = dur;

    particle(i).Best.Position = particle(i).Position;
    particle(i).Best.Cost = J;
    particle(i).Best.Duration = dur;

    if particle(i).Best.Cost < GlobalBest.Cost
        GlobalBest = particle(i).Best; %#ok<*NASGU>
    end
end

BestCost = nan(MaxIt,1);
BestDuration = nan(MaxIt,1);

%% 迭代
for it = 1:MaxIt
    for i = 1:nPop
        % 更新速度
        r1 = rand(1, nVar);
        r2 = rand(1, nVar);
        particle(i).Velocity = w*particle(i).Velocity ...
            + c1*r1.*(particle(i).Best.Position - particle(i).Position) ...
            + c2*r2.*(GlobalBest.Position - particle(i).Position);

        % 限幅速度
        particle(i).Velocity = max(particle(i).Velocity, VelMin);
        particle(i).Velocity = min(particle(i).Velocity, VelMax);

        % 更新位置
        particle(i).Position = particle(i).Position + particle(i).Velocity;

        % 位置越界处理（饱和 + 角度环绕）
        particle(i).Position = min(max(particle(i).Position, VarMin), VarMax);
        particle(i).Position(2) = wrapToPi(particle(i).Position(2)); % theta wrap
        % 强可行修复：保证 t0+t1 <= T_SUM_MAX（按比例缩放两段时间）
        t0i = particle(i).Position(3); t1i = particle(i).Position(4);
        tsum = t0i + t1i;
        if tsum > T_SUM_MAX
            sf = T_SUM_MAX / tsum; particle(i).Position(3:4) = particle(i).Position(3:4) * sf;
        end

        % 重新计算适应度
        [J, dur] = obj(particle(i).Position);
        particle(i).Cost = J;
        particle(i).Duration = dur;

        % 更新个体最优
        if J < particle(i).Best.Cost
            particle(i).Best.Position = particle(i).Position;
            particle(i).Best.Cost = J;
            particle(i).Best.Duration = dur;
        end

        % 更新全局最优
        if particle(i).Best.Cost < GlobalBest.Cost
            GlobalBest = particle(i).Best;
        end
    end

    BestCost(it) = GlobalBest.Cost;
    BestDuration(it) = GlobalBest.Duration;

    % 衰减惯性权重
    w = w * wDamp;

    % 迭代日志
    fprintf('Iter %3d | BestDuration = %.6f s | Cost = %.6f\n', it, GlobalBest.Duration, GlobalBest.Cost);
end

%% 输出最优解（将 s,theta 转为 vx,vy）
s_best = GlobalBest.Position(1);
th_best = GlobalBest.Position(2);
t0_best = GlobalBest.Position(3);
t1_best = GlobalBest.Position(4);

vx_best = s_best * cos(th_best);
vy_best = s_best * sin(th_best);

fprintf('\n最优解：\n');
fprintf('  s (speed) = %.6f m/s\n', s_best);
fprintf('  theta (rad) = %.6f\n', th_best);
fprintf('  vx_FY1 = %.6f m/s\n', vx_best);
fprintf('  vy_FY1 = %.6f m/s\n', vy_best);
fprintf('  t_throw = %.6f s\n', t0_best);
fprintf('  t_explode = %.6f s\n', t1_best);

[bestTotal, bestIntervals] = q1_occlusion_time(vx_best, vy_best, t0_best, t1_best, q1_opts);
fprintf('  最优遮挡总时长 = %.6f s\n', bestTotal);
if ~isempty(bestIntervals)
    disp('  遮挡区间 [t_begin, t_end] (s):');
    disp(bestIntervals);
end

%% 可视化收敛曲线
figure('Name','PSO 收敛曲线');
subplot(2,1,1); plot(BestDuration,'LineWidth',1.5); grid on;
ylabel('Best Duration (s)'); xlabel('Iteration'); title('遮挡时长');
subplot(2,1,2); plot(BestCost,'LineWidth',1.5); grid on;
ylabel('Best Cost'); xlabel('Iteration'); title('目标函数值');

%% 目标函数：从 [s,theta,t0,t1] 计算遮挡（最小化 -duration）
function [J, duration] = objective_s_theta(x, q1_opts)
    s = x(1); th = x(2); t0 = x(3); t1 = x(4);
    vx = s * cos(th); vy = s * sin(th);
    try
        [duration, ~] = q1_occlusion_time(vx, vy, t0, t1, q1_opts);
    catch
        duration = 0;
    end
    J = -duration;
end

%% von Mises 随机采样（Best & Fisher 算法）
function theta = randVonMises(mu, kappa, n)
    if kappa < 1e-8
        theta = (rand(n,1)*2*pi - pi) + mu; % 近似均匀
        theta = mod(theta + pi, 2*pi) - pi;
        return;
    end
    a = 1 + sqrt(1 + 4*kappa^2);
    b = (a - sqrt(2*a)) / (2*kappa);
    r = (1 + b^2) / (2*b);
    theta = zeros(n,1);
    for ii = 1:n
        while true
            u1 = rand; u2 = rand; u3 = rand;
            z = cos(pi*u1);
            f = (1 + r*z) / (r + z);
            c = kappa * (r - f);
            if (u2 < c*(2 - c)) || (log(u2) <= (c - 1))
                break;
            end
        end
        signv = 1;
        if u3 - 0.5 < 0, signv = -1; end
        theta(ii) = mu + signv * acos(f);
    end
    theta = mod(theta + pi, 2*pi) - pi; % wrap到[-pi,pi]
end 