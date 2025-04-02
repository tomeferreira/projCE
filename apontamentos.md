# Apontamentos

- Começar por experimentar evoluir logo a matriz de ints (tipo a do robot_structure do random_controler)

- As runs são deterministicas (desde que a seed seja a mesma)

- Usar GA como baseline, em vez de random. No entanto não percebi bem se o stor gostou da ideia ou não.

## Ambiente de programação

Em adição aos requisitos dados, foi adicionado o pandas para guardar os dados de cada run.

## Evolve structure

### Fitness

A função de fitness é semelhante à dada para o `random_structure`, com apenas uma alteração. Foi adicionado uma penalização caso o robot não esteja todo conectado.

O professor disse que esta penalização não é a melhor solução, pois muitos robots, após a mutação, podem não estar conectados. Isto pode levar a avg fitness a baixar muito, e a evolução não vai ser tão boa. Uma solução melhor seria fazer com que o robot nunca ficasse desconectado.

## Evolve controller

A primeira população, que é gerada random, a sua fitness não está a ser guardada nos resultados

### Fitness

A função de fitness é semelhante à dada para o `random_controller`.

### Resultados

Eu (Miguel) decidi usar uma abordagem diferente da do Tomé no momento de guardar resultados. O dataframe pandas não é muito bom para conseguir guardar os parâmetros usados naquela experiencia. Decidi usar um ficheiro txt normal para conseguir guardar os parametros no inicio, e depois o resultado das experiencias
