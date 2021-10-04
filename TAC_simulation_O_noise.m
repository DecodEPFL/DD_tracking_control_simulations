clear

%% Model parameters
% parameters are randomly selected, the only 
% consideration is that A cannot have large eigenvalues
% otherwise,Yp Yf would be large and the optimization solver has numerical issues

n=4;
m=3;
p=2;

% A=2*rand(n,n)-1;

A = [0.6799   -0.0331   -0.8332    0.4924;
     0.9748    1.0060    0.3666    0.5863;
     0.7311    0.3693   -1.0711    0.1603;
    -0.7442    0.0330    0.0667    0.1961];

% B=2*rand(n,m)-1;

B = [-0.7841   -0.1798   -0.0757;
      0.5204   -0.5806   -0.6510;
      0.1974    0.2140   -0.4851;
     -0.9378    0.7881   -0.1826];

% C=2*rand(p,n)-1;

C = [0.4458    0.4911    0.7394   -0.1359
     0.0733   -0.1468   -0.6357    0.7353];

maxeig = max(abs(eig(A)))

% A = A/maxeig*1.0173;

% if maxeig>1.05
%     A = A/maxeig*(0.899+0.15*rand);
% end

max(abs(eig(A)))

% while true
%% Generate noiseless Up Yp Uf Yf
Td=110;   % Number of historical data points
Te=20;    % Horizon length
Tini = n+2;   % Tini > lag(system) should hold
% x=zeros(3,T+1);
x0 = rand(n,1);
u=1*(2*rand(m*Td,1)-1);


% y=zeros(p*T,1);

O = zeros(n*Td,n);
H = zeros(n*Td,m*Td);
HC = zeros(p*Td,m*Td);
for t = 1:Td
    O((t-1)*n+1:t*n,:) = A^(t-1);
    for k = 1:t-1
        H((t-1)*n+1:t*n,(k-1)*m+1:k*m) = A^(t-k-1)*B;
        HC((t-1)*p+1:t*p,(k-1)*m+1:k*m) = C*A^(t-k-1)*B;
    end
end

OC = kron(eye(Td),C)*O;
% HC
 
% for t=1:T
%     x(:,t+1)=A*x(:,t)+B*u(t);
%     y(t)=C*x(:,t);
% end

x = O*x0+H*u;
y = OC*x0+HC*u;

% construct hankel matrix from u and y
Utotal=zeros((Tini+Te)*m,Td-(Tini+Te)+1);
Ytotal=zeros((Tini+Te)*p,Td-(Tini+Te)+1);
for i=1:size(Utotal,2)
    Utotal(:,i)=u((i-1)*m+1:(i-1)*m+(Tini+Te)*m);
    Ytotal(:,i)=y((i-1)*p+1:(i-1)*p+(Tini+Te)*p);
%     Utotal(i,:)=u(i:30+i);
%     Ytotal(i,:)=y(i:30+i);
end

% Up,Yp,Uf,Yf 
Up=Utotal(1:Tini*m,:);
Yp=Ytotal(1:Tini*p,:);
Uf=Utotal(Tini*m+1:end,:);
Yf=Ytotal(Tini*p+1:end,:);

%% Generate noisy uini yini
%  WARNING: a
%  uini -> the input sequence
%  yini -> the NOISY output sequence (what we use in calculations)


% bounded measurement noise rho with \|rho\|^2\le \epsilon_rho

epsilon=0.001;
rho=2*sqrt(epsilon)*rand(Tini*p,1)-sqrt(epsilon);

% x=zeros(3,6);
uini=2*rand(Tini*m,1)-1;  % Input 
% y=zeros(n*p,1);

Os = O(1:Tini*n,:);
Hs = H(1:Tini*n,1:Tini*m);
OCs = OC(1:Tini*p,:);
HCs = HC(1:Tini*p,1:Tini*m);

xini=rand(n,1);
load('exp1_init.mat')
% x(:,1)=xini;

% for t=1:5
%     x(:,t+1)=A*x(:,t)+B*u(t);
%     y(t)=C*x(:,t)+w(t);
% end

x = Os*xini + Hs*(uini);
yini = OCs*xini + HCs*(uini) + rho;
% yini == kron(eye(Tini),C)*x+rho

%% Feasible Noise Parameterization

Phi11=Tini*p*epsilon;
Phi12=zeros(1,Tini*p);
Phi21=Phi12';
Phi22=-eye(Tini*p);

Ncal=null(Up);
M = Ncal*orth(Ncal'*Yp');

gw_star = Up'*inv(Up*Up')*uini;

w0=-Yp*gw_star+yini;

Aw_11=Phi11+w0'*Phi12'+Phi12*w0+w0'*Phi22*w0;
Aw_12=-Phi12*Yp*M-w0'*Phi22*Yp*M;
Aw_21=Aw_12';
Aw_22=M'*Yp'*Phi22*Yp*M;

Aw=[Aw_11,Aw_12;
    Aw_21,Aw_22];

%% Performance specification transformation

%% calculate Yp1 such that there exists a permutation matrix PY giving
% PY*Yp=[Yp1;Yp2]
% with [Up;Yp1;Uf] full row rank;
% and rank([Up;Yp1;Uf])=rank([Up;Yp;Uf])

Yp1 = [];
Yp2 = Yp;

while 1 
   
    for i = 1:size(Yp2,1)
        if rank([Up;Yp1;Yp2(i,:);Uf])>rank([Up;Yp1;Uf])
            Yp1 = [Yp1;Yp2(i,:)];
            Yp2 = [Yp2(1:i-1,:);Yp2(i+1:end,:)];
            break
        end 
    end
    
    if rank([Up;Yp1;Uf])==rank([Up;Yp;Uf])
        break
    end 
end

%% calculate Bini, Bu, Bw, and y0

Lambda = [Up;Yp1;Uf];

Bini = Yf*(eye(size(Yf,2))+Lambda'*inv(Lambda*Lambda')*[zeros(size(Lambda,1)-Te*m,size(Uf,2));-Uf]);
Bu = Yf*Lambda'*inv(Lambda*Lambda')*[zeros(size(Lambda,1)-Te*m,Te*m);eye(Te*m)];

y0 = Bini*gw_star;
Bw = Bini*M;

%% solve the optimization problem
% min_{alpha,u,gamma} gamma
% s.t., finalLMI>=0

ops = sdpsettings('solver','mosek'); 
ops = sdpsettings(ops,'verbose',0);
alpha=sdpvar(1);
optu=sdpvar(Te*m,1);
gamma=sdpvar(1,1);

% generate matrix Q_g^a
r=zeros(Te*p,1);
barR=eye(Te*m);
barQ=eye(Te*p);

Qga_11=gamma-(y0-r)'*barQ*Bu*optu-optu'*Bu'*barQ*(y0-r)-(y0-r)'*barQ*(y0-r);
Qga_12=-(y0+Bu*optu-r)'*barQ*Bw;
Qga_21=Qga_12';
Qga_22=-Bw'*barQ*Bw;
Qga=[Qga_11,Qga_12
     Qga_21,Qga_22];

% generate final LMI matrix

tempSize=size(barR+Bu'*barQ*Bu,1);
zeroBlock=zeros(tempSize,size(Qga,2)-1);
finalLMI=[inv(barR+Bu'*barQ*Bu), optu, zeroBlock;
          [optu';zeroBlock'],Qga-alpha*Aw];


CONS1=[alpha>=0];
CONS2=[finalLMI>=0];
CONS=[CONS1,CONS2];

infosold = optimize(CONS,gamma*gamma,ops);
desiredControlInput = double(optu);
desiredGamma=double(gamma);

if infosold.problem ~=0
    disp('Controller cannot be computed!')
   return 
end

%% verification
% check that: using the above obtained desiredControlInput
% the performance guarantees for noiseless system can be obtained

optu=desiredControlInput;

% OCt = OC(p+1:(N+1)*p,:);
% HCt = HC(p+1:(N+1)*p,m+1:(N+1)*m);
% 
% xinit = x((Tini-1)*n+1:Tini*n);
% 
% opty=OCt*xinit+HCt*optu;
% 
% % x=zeros(3,N+1);
% % x(:,1) = xini;
% % opty=zeros(N,1);
% % for t=1:N
% %     x(:,t+1)=A*x(:,t)+B*optu(t);
% %     opty(t)=C*x(:,t+1);
% % end

OCt = OC(1:(Tini+Te)*p,:);
HCt = HC(1:(Tini+Te)*p,1:(Tini+Te)*m);

% x = Ot*xini + Ht*[uini;optu];
opty = OCt*xini + HCt*[uini;optu];
opty = opty(Tini*p+1:end);

figure
for j = 1:p
   subplot(p,1,j)
   plot(opty(j:p:end));
end

noiselessGamma=optu'*barR*optu+(opty-r)'*barQ*(opty-r);
      
% if ~((optu'*optu<=8) && (opty'*opty<=7))
%     break
% %     return
% else 
%     close all
% end
% 
% end
%% randomly select a feasibility noise and check whether 
% the LQ performance is smaller than the previous obtained one

% generate feasible g0 and further obtain noise w
optg=sdpvar(size(Aw,1)-1,1);

CONS1=[[1;optg]'*Aw*[1;optg]>=1e-3];
CONS = [CONS1];

infosolt = optimize(CONS,optg'*optg,ops);
if infosolt.problem ~= 0
    disp('No feasible gbar found!')
    return
end
desiredg = double(optg);

N_noisesamples = 100;
y_s = zeros(Te*p,N_noisesamples);
w_s = zeros(Tini*p,N_noisesamples);
gammas = zeros(N_noisesamples,1);

for indx = 1:N_noisesamples
   
    while 1
        desiredgn = desiredg + 0.05*(2*rand(size(desiredg))-1);
        consw = [1;desiredgn]'*Aw*[1;desiredgn];
        if (consw>=0)
            break;
        end
    end
    
    candidate_w=-Yp*M*desiredgn+w0;
    
    w_s(:,indx) = candidate_w;
    
    ops = sdpsettings('solver','mosek'); 
    ops = sdpsettings(ops,'verbose',0);
    % optu=sdpvar(N*m,1);
    opty=sdpvar(Te*p,1);
    optg=sdpvar(size(Up,2),1);

    % generate matrix B8
    r=zeros(Te*p,1);
    barR=eye(Te*m);
    barQ=eye(Te*p);

    % generate final LMI matrix
    obj=(optu)'*barR*(optu)+(opty-r)'*barQ*(opty-r);

    CONSF=[[Up;Yp;Uf;Yf]*optg==[uini;yini-candidate_w;optu;opty]];

    infosol = optimize(CONSF,[],ops);
    double(obj)
    gammas(indx) = double(obj);
    y_s(:,indx) = double(opty);
end


% Plot the output trajectories
int = 'Interpreter';
lat = 'Latex';
fonts = 'Fontsize';
fs = 14;
font = 'FontName';
fn = 'Times';
line = 'LineWidth';
Line = 2.5;

figure(2)
set(gca, font, fn, fonts, fs);
hold on
for j = 1:p
   subplot(p,1,j)
   plot(y_s(j:p:end,:));
   grid on
   ylabel(['$$\mathbf{y}_{',num2str(j),'}(k)$$'], font, fn, int,lat,fonts,fs)
   xlim([1 Te])
end

xlabel('$$k$$', font, fn, int,lat,fonts,fs)

% figure(4)
% set(gca, font, fn, fonts, fs);
% subplot(3,1,1)
% hold on
% scatter(1:N_noisesamples,gammas,'b',line,Line);
% plot([1 N_noisesamples],[desiredGamma desiredGamma],'r',line,Line);
% grid on
% ylabel(['$$\mathbf{\gamma}$$'], font, fn, int,lat,fonts,fs)
% xlim([1 N_noisesamples])
% subplot(3,1,2)
% hold on
% scatter(1:N_noisesamples,inconsts,'b',line,Line);
% % plot([1 N_noisesamples],[N*epsilon_u N*epsilon_u],'r',line,Line);
% grid on
% ylabel(['$$\mathbf{\psi(\bar{u})}$$'], font, fn, int,lat,fonts,fs)
% xlim([1 N_noisesamples])
% subplot(3,1,3)
% hold on
% scatter(1:N_noisesamples,outconsts,'b',line,Line);
% % plot([1 N_noisesamples],[N*epsilon_y N*epsilon_y],'r',line,Line);
% grid on
% ylabel(['$$\mathbf{\theta(y)}$$'], font, fn, int,lat,fonts,fs)
% xlim([1 N_noisesamples])
% 
% xlabel('Number of noise and disturbance sample', font, fn, int,lat,fonts,fs)

figure(5)
set(gca, font, fn, fonts, fs);
hold on
scatter(1:N_noisesamples,gammas,'b',line,Line);
plot([1 N_noisesamples],[desiredGamma desiredGamma],'r',line,Line);
grid on
ylabel(['$$\mathbf{\gamma}$$'], font, fn, int,lat,fonts,fs)
xlim([1 N_noisesamples])
xlabel('Number of noise and disturbance sample', font, fn, int,lat,fonts,fs)

all(gammas<desiredGamma)



desiredGamma
% trueGamma
noiselessGamma
infosold
% infosolt
max(abs(eig(A)))
noiselessGamma<desiredGamma
% trueGamma<desiredGamma
