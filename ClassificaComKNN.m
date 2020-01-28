%Autor: Denilson Gomes Vaz da Silva <denilsongomes@alu.ufc.br>
%Graduando em Engenharia da Computação
%Reconhecimento de padroes

%Algoritmo para classificar base de dados usando o K-NN
%Os valores para kfold e kvizinhos devem ser alterado manualemente
clc
close all
clear

%Carrega base de dados
load('Classe1.mat')
load('Classe2.mat')
Y1 = Classe1';
Y2 = Classe2';

%Plotagem dos dados
figure, plot(Y1, 'r')
hold on
plot(Y2, 'b')
hold off

%Normaliza os sinais no tempo (media = 0 e variança = 1)
for i=1:50
    Y1(i,:) = (Y1(i,:) - mean(Y1(i,:)))/std(Y1(i,:));
    Y2(i,:) = (Y2(i,:) - mean(Y2(i,:)))/std(Y2(i,:));
end

%Constroi o dataset
dataset = ones(100,501);
dataset(1:50,1:end-1) = Y1;
dataset(51:end,1:end-1) = Y2;
%atribui o rotulo de cada classe2
dataset(51:end,end) = 2;

%Estudo dos dados

% Calcular Kurtoise e Skewness no tempo
for i=1:100   
    skew(i) = (sum((dataset(i,:)-mean(dataset(i,:))).^3)./length(dataset(i,:)))./ (var(dataset(i,:),1).^1.5);
    kurt(i) = (sum((dataset(i,:)-mean(dataset(i,:))).^4)./length(dataset(i,:))) ./ (var(dataset(i,:),1).^2);
end
% Plotar skewness e Kurtoise das duas classes
% Skewness
figure, plot(skew(1:50), 'b')
hold on
plot(skew(51:100), 'r')
hold off
title('Skewness')

% Kurtoise
figure, plot(kurt(1:50), 'b')
hold on
plot(kurt(51:100), 'r')
hold off
title('Kurtoise')

%Amplitude Maxima do sinal no tempo
%Amplitude Media do sinal no tempo
%Desvio Padrão do sinal no tempo
for i=1:100
    AmpMax(i) = max(dataset(i,1:500));
    AmpMedia(i) = mean(dataset(i,1:500));
    DesvioAmp(i) = std(dataset(i,1:500));
end
% %Atributos da classe 1
% disp('Atributos da classe 1')
% AmpMax1 = mean(AmpMax(1:50))
% AmpMedia1 = mean(AmpMedia(1:50))
% DesvioAmp1 = mean(DesvioAmp(1:50))
% 
% %Atributos da classe 2
% disp('Atributos da classe 2')
% AmpMax2 = mean(AmpMax(51:end))
% AmpMedia1 = mean(AmpMedia(51:end))
% DesvioAmp1 = mean(DesvioAmp(51:end))

%TF do sinal
for i=1:100   
    YTF(i,1:500)=(abs(fftshift(fft(dataset(i,1:500)))));
end
freq_vec = linspace(-pi,pi,length(dataset(1,1:500)));
% figure,plot(freq_vec,abs(YTF(1,1:500)),'b')
% figure,plot(freq_vec,abs(YTF(62,1:500)),'b')

%Kurtoise
%Skewness
%maximo Modulo da TF
%media
%mediana
for i=1:100   
    ModuloFurrier(i) = max(abs(YTF(i,:)));
    skew(i) = (sum((YTF(i,:)-mean(YTF(i,:))).^3)./length(YTF(i,:)))./ (var(YTF(i,:),1).^1.5);
    kurt(i) = (sum((YTF(i,:)-mean(YTF(i,:))).^4)./length(YTF(i,:))) ./ (var(YTF(i,:),1).^2);
    if (i <=50)
        MediaFurrier1(i) = mean(abs(YTF(i,1:50)));
        MedianaFurrier1(i) = median(abs(YTF(i,1:50)));
    end
    if (i>50)
        MediaFurrier2(i) = mean(abs(YTF(i,51:100)));
        MedianaFurrier2(i) = median(abs(YTF(i,51:100)));
    end
end
% MaxModuloFurrier1 = mean(ModuloFurrier(1:50))
% MaxModuloFurrier2 = mean(ModuloFurrier(51:end))
% 
% MediaFurrier1 = mean(MediaFurrier(1:50))
% MediaFurrier2 = mean(MediaFurrier(51:end))
% 
% MedianaFurrier1 = mean(MedianaFurrier(1:50))
% MedianaFurrier2 = mean(MedianaFurrier(51:end))

%Estes atributos serão usados na classificacao
%NO TEMPO {%Kurtoise, Skewness}
%Logo, vamos construir nossa base de dados
dados(:,1) = kurt;
% dados(:,1) = skew;
% dados(:,3) = DesvioAmp/max(DesvioAmp);
% dados(:,4) = kurt/max(kurt);
% dados(:,5) = skew/max(skew);
% dados(:,6) = ModuloFurrier/max(ModuloFurrier);
% dados(1:50,7) = MediaFurrier1/max(MediaFurrier1);
% dados(51:100,7) = MediaFurrier2(51:end)/max(MediaFurrier2(51:end));
% dados(1:50,8) = MedianaFurrier1/max(MedianaFurrier1);
% dados(51:100,8) = MedianaFurrier2(51:end)/max(MediaFurrier2(51:end));
classes(1:50) = dataset(1:50,end);
classes(51:100) = dataset(51:end,end);

%k-Fold (para k=5 e k=10)
kfold=5;
kvizinhos = 3;
disp('Matriz confusão da classificação do KNN com K-Fold')
for i=1:kfold
    [~,numAtributos] = size(dados);
    acertos1 = 0;
    erros1 = 0;
    acertos2 = 0;
    erros2 = 0;
    for t=1:length(dados)/kfold %asmostras (Testes)
        dadosAux = dados;
        classeAux = classes;
        %pega as amostras de teste
        Teste(1:length(dados)/(2*kfold),:)=dadosAux(1+(i-1)*length(dados)/(2*kfold):i*length(dados)/(2*kfold),:);
        classeTeste(1:length(dados)/(2*kfold))=classes(1+(i-1)*length(dados)/(2*kfold):i*length(dados)/(2*kfold));
        Teste(length(dados)/(2*kfold)+1:length(dados)/kfold,:)=dadosAux(51+(i-1)*length(dados)/(2*kfold):50+i*length(dados)/(2*kfold),:);
        classeTeste(length(dados)/(2*kfold)+1:length(dados)/kfold)=classes(51+(i-1)*length(dados)/(2*kfold):50+i*length(dados)/(2*kfold));
    
        %pega as amostras de treino
        dadosAux(1+(i-1)*length(dados)/(2*kfold):i*length(dados)/(2*kfold),:)=[];
        classeAux(1+(i-1)*length(dados)/(2*kfold):i*length(dados)/(2*kfold))=[];
        dadosAux(51+(i-1)*length(dados)/(2*kfold)-length(dados)/(2*kfold):50+i*length(dados)/(2*kfold)-length(dados)/(2*kfold),:)=[];
        classeAux(51+(i-1)*length(dados)/(2*kfold)-length(dados)/(2*kfold):50+i*length(dados)/(2*kfold)-100/(2*kfold))=[];
    
        for j=1:length(dadosAux) %asmostras (Treinamento)
            d=0; %reseta d para a distancia de um novo vetor
            for p=1:numAtributos %para todos os atributos das amostras
                %d é a soma dos quadrados das diferencas das coordenadas da amostra i para a amostra j
                d = d + (Teste(t,p) - dadosAux(j,p))^2; 
            end
            %dist é a raiz quadrada da soma dos quadrados das coordenadas (distancia euclidiana)
            dist(j) = sqrt(d);
        end
        
        for m=1:kvizinhos %para os k-vizinhos mais perto da amostra t
            [~,indice] = min(dist); %pegamos o indice do vizinho mais proximo
            knn(m) = classeAux(indice); %colocamos a classe desta amostra em knn
            dist(indice) = []; %removemos esta amostra
            dadosAux(indice,:) = []; %removemos esta amostra
            classeAux(indice) = []; %removemos a classe da amostra retirada (questao de dimensionamento)
        end
        %verificamos se a classe que ocorre em maior numero eh a classe
        %real do teste
        if(mode(knn) == classeTeste(t) && classeTeste(t) == 1) %caso acerte um teste da classe real 1
            acertos1 = acertos1 + 1; %acrescentamos acertos1
        end
        if(mode(knn) == classeTeste(t) && classeTeste(t) == 2) %caso acerte um teste da classe real 2
            acertos2 = acertos2 + 1; %acrescentamos acertos2
        end
        if(mode(knn) ~= classeTeste(t) && classeTeste(t) == 1) %caso erre um teste da classe real 1
            erros1 = erros1 + 1; %acrescentamos erros1
        end
        if(mode(knn) ~= classeTeste(t) && classeTeste(t) == 2) %caso erre um teste da classe real 2
            erros2 = erros2 + 1; %acrescentamos erros2
        end
    end
    %Matriz de confusão: 
    %Elemento 11 mostra o numero de vez que o algoritmo disse ser a classe 1, e de fato era a classe 1.
    %Elemento 12 mostra o numero de vez que o algoritmo disse ser a classe 2, e era a classe 1.
    %Elemento 21 mostra o numero de vez que o algoritmo disse ser a classe 1, e era a classe 2.
    %Elemento 22 mostra o numero de vez que o algoritmo disse ser a classe 2, e de fato era a classe 2.
    [acertos1 erros1; erros2 acertos2]
end

%leave-one-out
kvizinhos = 3;
acertos1 = 0;
erros1 = 0;
acertos2 = 0;
erros2 = 0;
disp('Matriz confusão da classificação do KNN com leave-on-out')
for t=1:length(dados) %asmostras (Testes)
    dadosAux = dados;
    classeAux = classes;
    
    %pega a amostra de teste
    Teste = dadosAux(t,:);
    classeTeste = classes(t);
         
    %pega as amostras de treino (retira a de teste)
    dadosAux(t,:)=[];
    classeAux(t)=[];
    for j=1:length(dadosAux) %asmostras (Treinamento)
        d=0; %reseta d para a distancia de um novo vetor
        for p=1:numAtributos %para todos os atributos das amostras
            %d é a soma dos quadrados das diferencas das coordenadas da amostra i para a amostra j
            d = d + (Teste(p) - dadosAux(j,p))^2; 
        end
        %dist é a raiz quadrada da soma dos quadrados das coordenadas (distancia euclidiana)
        dist(j) = sqrt(d);
     end
        
     for m=1:kvizinhos %para os k-vizinhos mais perto da amostra t
        [~,indice] = min(dist); %pegamos o indice do vizinho mais proximo
        knn(m) = classeAux(indice); %colocamos a classe desta amostra em knn
        dist(indice) = []; %removemos esta distancia
        dadosAux(indice,:) = []; %removemos esta amostra
        classeAux(indice) = []; %removemos a classe da amostra retirada (questao de dimensionamento)
     end
     %verificamos se a classe que ocorre em maior numero eh a classe real do teste
     if(mode(knn) == classeTeste && classeTeste == 1) %caso acerte um teste da classe real 1
        acertos1 = acertos1 + 1; %acrescentamos acertos1
     end
     if(mode(knn) == classeTeste && classeTeste == 2) %caso acerte um teste da classe real 2
        acertos2 = acertos2 + 1; %acrescentamos acertos2
     end
     if(mode(knn) ~= classeTeste && classeTeste == 1) %caso erre um teste da classe real 1
        erros1 = erros1 + 1; %acrescentamos erros1
     end
     if(mode(knn) ~= classeTeste && classeTeste == 2) %caso erre um teste da classe real 2
        erros2 = erros2 + 1; %acrescentamos erros2
     end

    %Matriz de confusão: 
    %Elemento 11 mostra o numero de vez que o algoritmo disse ser a classe 1, e de fato era a classe 1.
    %Elemento 12 mostra o numero de vez que o algoritmo disse ser a classe 2, e era a classe 1.
    %Elemento 21 mostra o numero de vez que o algoritmo disse ser a classe 1, e era a classe 2.
    %Elemento 22 mostra o numero de vez que o algoritmo disse ser a classe 2, e de fato era a classe 2.
end
[acertos1 erros1; erros2 acertos2]