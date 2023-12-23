clc
close all

%makenewnetwork=true; % true - make new network connectivity, false - use previous connectivity
                      % If false, either have previous data on worskpace or
                      % load a previous .mat file.
                      
savedata=true;        % saveworkspace - can be used in subsequent run
rec_spikes=true;

runtime=5000; %Total simulation?t2 time in (ms)
dt=0.2; %0.25 is the minimal
totalsteps=runtime/dt; %20,000
t=0:dt:runtime; %computed time points

%Network Structure::
N=1000; %number of neurons
sparseness=0.15; %Sparseness of the network and input connections

%Input Parameters
pinput=15 % 15 + jext=0.05 set of input parameters generate highly variable initial stimulus
jext=0.037; %Input weight strength
t1=0; %Input start time
t2=100; %Input end time

%Weights::
jrec_arr=[0.0, 0.0003, 0.0004, .00046, .00048, 0.000492, 0.000494]



%Time Constant 
Tm=20; %ms Membrane time constant
Tr=50; %ms Firing rate calculation window
Ts=20; %ms Synaptic activation time constant
%Tp=5000; %ms Time constant for Proto-weights


%Ican Parameters
%gmax=0.0135; %Maximum conductance-  1/oh=Siemens
%with these parameters the sustained firing rate is quite high (~40Hz)
gmax=0.0134; 
tht=1.0;
tethaCa=tht*ones(N,1);
ro=0.75; %0.787; %Ca input amount during one spike
Tca=100; %ms Time constant for the Ca channels
nh=4; %hill coefficient

%Membrane Voltage Parameters:
Vr=0; %mv Reversal potential
Vth=15; %mv Threshold potential
Ecan=80*ones(N,1); %mv 
gl=0.05; %milliSimemens Conductance of the leakage term
C=1; %microFarad  Capacitance | T=C/g=20ms
El=0*ones(N,1); %mV
Ee=55*ones(N,1); %mV
ros=1/7; 
R=250*ones(N,1); %ohm  Rezistance | g'=gcan.R






%Poisson input-Part-1
inputrate=pinput/1000; %spike/ ms
ratedt=dt*inputrate*ones(N,1);   %Hz

%if makenewnetwork
    %Input vector from previous network
    Lext=ones(N,1);
    %SPARSENESS FOR INPUT
    nonzero2 = sparseness*N*N;  %number of nonzero elements at the input vector
    Linput = zeros(N, N);        %Input matrix
    Linput(randperm(numel(Linput), nonzero2)) = 1; % Random sparse input matrix
    Linput(1:N+1:end) = 0; %set diagonal to zero

    % Recurrent Connections-Part     fgbb1 - Part 2
    nonzero = sparseness*N*N;  %number of nonzero elements at the recurrent matrix
    Lrec = zeros(N, N);        %Recurrent matrix
    Lrec(randperm(numel(Lrec), nonzero)) = 1; %Random sparse recurrent matrix
    Lrec(1:N+1:end) = 0; %set diagonal to zero

    %Lrec is the connection matrix
    %Ls is the weight matrix
    %Ls=jrec*ones(N);
%end

DECAY=[]; % Decay time array for each training 
MAXRATE=[]; % Max rate time array for each training 
WMean=[]; % Mean weight array for each training 

rec_ln=length(jrec_arr);

for kk=1:rec_ln  %20000  
    jrec=jrec_arr(kk)
    Ls=jrec*ones(N);
    
    %initial values:
    V=0.1*ones(N,1); % Voltage
    CA=zeros(N,1); % Calcium
    Rate=zeros(N,1); % Firing rate vector
    Sext=zeros(N,1); %Synaptic activation for external input
    Srec=zeros(N,1); %Synaptic activation for recurrent conections
    dirac=zeros(N,1); % Dirac vector for spikes
    dirac_arr=zeros(N,totalsteps); %matrix for storing spikes
   % H=zeros(N); %Hebbian
   % Lp=zeros(N); %Proto matrix
    
    
    Lconn=Ls.*Lrec; %Connection matrix
    
    Wmean=mean(mean(Lconn)); %Mean recurrent weights
    WMean=[WMean,Wmean]; %Recurrent mean weight array for each training
     
    for n=1:totalsteps
         
        %Poisson input- Part 2
        randi=rand(N,1);
        pre_spike1=ratedt(:)>randi;
        pre_spike=Linput*pre_spike1;
        
        %Transient input from excitatory populations::
        if t(n)>t1 && t(n)<t2
            
            pre_dirac=pre_spike;
            
        else pre_dirac=zeros(N,1);
            
        end
        
        %Ca influx at spiked neurons
        CA=CA+dt*(1/Tca)*(-CA)+ro*(dirac);
        gcan=gmax*((CA.^nh)./(CA.^nh+tethaCa.^nh));
        
        %Synaptic activation ::
        Sext=Sext+dt*(-(1/Ts)*Sext+ros*(ones(N,1)-Sext).*pre_dirac); %external input
        Srec=Srec+dt*(-(1/Ts)*Srec+ros*(ones(N,1)-Srec).*dirac); %recurrent connections
        
        %Voltagejext
        %V=V+dt*((1/C)*(gl*(El-V)+(Ls.*Lrec*Srec).*(Ee-V)+gcan.*(Ecan-V)+(jext*Lext.*Sext).*(Ee-V)));
        V=V+dt*((1/C)*(gl*(El-V)+(Lconn*Srec).*(Ee-V)+gcan.*(Ecan-V)+(jext*Lext.*Sext).*(Ee-V)));
        
        
        %when spikes:::
        V(V>Vth)=Vr;
        
        b=find(V==Vr);
       
        
        %Froming the dirac vector with spiked indices 1
        dirac=zeros(N,1);
        dirac(b)=1;
        
         if rec_spikes
             dirac_arr(:,n)=dirac;
         end
                
        %Calculate Firing rate
        Rate(:,n+1)=Rate(:,n)+(1/Tr)*(-dt*Rate(:,n)+(dirac));
        %Sext_mean_arr(n+1)=mean(Sext);
        Srec_mean_arr(n+1)=mean(Srec);
        
        %Learning Rule
        %H=1000*Rate(:,n+1)*1000*Rate(:,n+1)';
        %H(1:N+1:end) = 0; %set diagonal to zero
        %Lp=Lp+dt*(1/Tp)*(-Lp+H);
        %einh=beta*1000*(Rate(:,n+1));
  
%         if t(n)==trew
%             Leinh=repmat(einh,1,N);
%             AA = (Leinh < Rrew*2); % learning rule modification, so don't undershoot in correction
%             Rfunc = (Rrew*ones(N)-Leinh).*AA +(AA - (ones(N))*Rrew);
%             dLs=dt*nu*Lp.*Rfunc;            
%         end
        
    end
    

   % Ls=Ls+dLs; 
    
    %decay time
    %decay time
    meanrate=mean(1000*Rate);
    roundrate=round(meanrate);
    maxRate=max(roundrate);
    tarR=round(maxRate/exp(1));
    %indidecay=find(roundrate==Rrew);
    ind_t2=round(t2/dt);
    %indidecay=find(roundrate(ind_t2:end)==tarR,1);
    indidecay=find(roundrate(ind_t2:end)==10.0,1);
    if isempty(indidecay)
        indidecay=totalsteps-1;
        'bistable'
    else
        indidecay=indidecay+ind_t2;
    end
    decay=t(indidecay);
    decay=decay-t2
    DECAY=[DECAY;decay];
    
    %meanrate=mean(1000*Rate);
    %roundrate=round(meanrate,0);
    %indidecay=find(roundrate==Rrew);
    %decay=max(indidecay)*0.2;
    %DECAY=[DECAY;decay-t2];

    figure(1)
    %subplot(3,1,1)
    plot(t,meanrate,'linewidth',2)
    set(gca,'FontSize',12)
    title('Firing Rate', 'FontSize', 12)
    ylabel('Hz', 'FontSize', 12)
    xlabel('time(ms)', 'FontSize', 12)
    hold on
%     subplot(3,1,2)
%     plot(WMean,'*k','linewidth',2)
%     set(gca,'FontSize',12)
%     hold on
%     ylabel('J', 'FontSize', 12)
%     xlabel('tial#', 'FontSize', 12)
%     title('Mean Synaptic weight', 'FontSize', 12)
%     subplot(3,1,3)
%     plot(DECAY,'*k','linewidth',2)
%     set(gca,'FontSize',12)
%     hold on
%     title('Decay time', 'FontSize', 12)
%     ylabel('DecayTime', 'FontSize', 12)
%     xlabel('trial#', 'FontSize', 12)
%     
%     pause(0.001);
%     %figure(2)
%     %plot(t,Srec_mean_arr)
%     %hold on
end

%Plot last curve is differerent line type,
% figure(1)
% subplot(3,1,1)
% plot(t,meanrate,'k--','linewidth',2)
if rec_spikes
    figure(100)
    %imagesc(dirac_arr(200:300,:));
    imagesc(t,[200:300],dirac_arr(200:300,:))
    figure(200)
    %imagesc(t(15000:16000),[200:250],dirac_arr(200:250,15000:16000))
    imagesc(t(2000:3400),[200:250],dirac_arr(200:250,2000:3400))
end

figure(3)
plot(jrec_arr,DECAY,'x')
hold

% tp=fittype('a* (x0-x) .^eta ','coefficients',{'a','x0','eta'},'indep','x')
% f=fit(jrec_arr',DECAY,tp,'StartPoint',[0.0001,0.0012,-2])
% plot(jrec_arr,f);




if savedata
    Filename=['TA_',datestr(now)]; %do i need a .mat/.dat etc?
    save(Filename);
end



