import numpy as np
import random
import episiming.scenes.functions

class School:

    def __init__(self, scale, mtrx_school, pop_position, pop_age):
        self.scale = scale
        self.original_mtrx_school = mtrx_school
        self.mtrx_school = mtrx_school
        self.school_position = []
        self.dist_school_age_fraction = []
        self.dist_school_age = []
        self.students = []
        self.students_age = []
        self.pop_position = pop_position
        self.pop_age = pop_age

    def rescale_schools(self):
        """Função que faz a redução da quantidade de escolas em dado Heatmap

        Args:
            tx_reducao (float): A taxa de redução a qual esta submetido o cenário.
            mtrx_escolas (array): Array que contém as matrizes/heatmaps da distribuição das escolas por
            modadlidade, também pode ser passado somente uma matriz.

        Returns:
            array: Array de matrizes/heatmaps reduzidos
        """

        if self.scale == 1:
            if len(self.original_mtrx_school.shape) < 3:
                self.mtrx_school = np.array([self.original_mtrx_school])
            else:
                self.mtrx_school = self.original_mtrx_school
        else:
            ## Caso seja passado somente um heatmap, ao inves de uma lista de heatmaps
            ## Transforma em uma lista unitária de heatmap
            if len(self.original_mtrx_school.shape) < 3:
                self.original_mtrx_school = np.array([self.mtrx_school])
            num_school = np.sum(self.original_mtrx_school)
            rng = np.arange(np.prod(self.original_mtrx_school[0].shape))
            ## Para cada modalidade, a quantidade de escolas reduzidas
            num_type = map(lambda x: np.rint(np.sum(x) * self.scale).astype(int), self.original_mtrx_school)
            ## Faz uma escolha aleatória com pesos nos blocos do heatmap de cada modalidade
            weight_type = map(lambda x: (x / np.sum(x)).flatten(), self.original_mtrx_school)
            rescaled_school_type = map(lambda x, y: random.choices(rng, x, k=y), weight_type, num_type)

            ## Preenchemos, para cada modalidade, os blocos sorteados
            rescaled_mtrx = np.zeros(self.original_mtrx_school.shape)
            for i, m in enumerate(rescaled_school_type):
                m = np.array(m)
                row = np.floor(m / self.original_mtrx_school.shape[-1])
                col = np.mod(m, self.original_mtrx_school.shape[-1])
                for k, j in zip(row, col):
                    rescaled_mtrx[i][int(k)][j] += 1
            self.mtrx_school = rescaled_mtrx


    def distribute_age_school(self, school_age_groups, school_age_groups_fraction, max_age = 100):
        """Função que calcula a quantidade de matriculas por idade e modalidade escolar, segundo censo

         Args:
             idade_max (int): Idade max da população do cenário
             grupos_idade_escolar (array): Array com a separação de idades por grupo, onde cada par de elementos
             representa um intervalo do tipo [a,b)
             dist_grupos_idade (array): Array com a distribuição de cada grupo de idades, segundo censo.
             Pode ser passado apenas um array com a distribuição, ou um array de arrays com distribuição
             que representa a distribuição por modalidades

         Returns:
             array: Array com matrizes de tamanho 100, representando a quantidade total de cada idade de aluno em
             uma modalidade
         """

        ## Caso seja passado somente um array em dist_grupos_idade
        if len(school_age_groups_fraction.shape) < 2:
            school_age_groups_fraction = np.array([school_age_groups_fraction])

        ## Matriz com a distribuição de idade, separado idade a idade
        dist_school_age_fraction = np.zeros((len(school_age_groups_fraction), max_age))

        ## Transforma a informação de distribuição por grupo de idade para distribuição por idade
        for j in range(len(school_age_groups_fraction)):
            for i in range(len(school_age_groups)):
                if i < len(school_age_groups_fraction[j]):
                    dist_school_age_fraction[j][school_age_groups[i]:school_age_groups[i + 1]] = school_age_groups_fraction[j][i]

        ## Calcula a quantidade de alunos na idade, por modalidade
        dist_pop_age = np.array([np.count_nonzero(self.pop_age == i) for i in range(max_age)])
        self.dist_school_age_fraction = dist_school_age_fraction
        self.dist_school_age = np.round(dist_school_age_fraction[:school_age_groups[-2]] * dist_pop_age)


    def select_students(self):
        """Função que escolhe uma quantidade de individuos na população, pela idade, para serem matriculados
        em alguma escola

        Args:
            pop_idades (list(int)): Uma lista da idade de cada invididuo na população
            idades_escola (list(int)): Uma lista com a distribuição de alunos por idade

        Returns:
            array: Array com os indices dos alunos na população, separados pela distribuição de idade
        """
        enroll_type = []
        for j in range(len(self.dist_school_age)):
            students = np.array(
                [np.random.permutation(np.arange(len(self.pop_age))[self.pop_age == i])[:int(self.dist_school_age[j][i])]
                 for i in range(len(self.dist_school_age[j]))])
            enroll_type.append(students)
        self.students_age = np.array(enroll_type)


    def gen_school_position(self):
        """Função que gera as escolas, indexando pela sua posição

            Args:
                mtrx_escolas (array): Lista com heatmaps de escolas por modalidade

            Returns:
                array: Lista com a posição de cada escola, por modalidade
        """
        school_position = []
        for m in self.mtrx_school:
            rng = np.arange(np.prod(m.shape))
            num_school_per_block = m[m > 0].astype(int)
            pos = rng[m.flatten() > 0]
            aux = []
            for p, q in zip(pos, num_school_per_block):
                for i in range(q):
                    aux.append((np.mod(p, m.shape[1]), p // m.shape[1]))
            school_position.append(aux)
        self.school_position = school_position


    def alloc_students(self):
        """Função que aloca um aluno em uma das 3 escolas mais próximas de sua posição

        Args:
            alunos (array): Array com os indices dos alunos na população, separados pela distribuição de idade,
            este input é o output da função escolhe_aluno_idades
            pos_escolas (array): Lista com a posição de cada escola, por modalidade, este input é o
            output da função gera_pos_escolas

        Returns:
            array: Indice de alunos na população, separados por modalidade de escola
        """

        enroll_type = [[[] for _ in range(len(p))] for p in self.school_position]
        for i in range(len(self.students_age)):
            student_type = np.hstack(self.students_age[i])
            student_position = np.array(self.pop_position)[student_type] * 10
            dist_student = [np.linalg.norm(p_i - self.school_position[i], axis=1).argsort()[:6] for p_i in
                            student_position if len(self.school_position[i]) > 0]
            enroll = [random.choices(k)[0] for k in dist_student]
            for k in range(len(enroll)):
                enroll_type[i][enroll[k]].append(student_type[k])

        self.students = np.array(enroll_type)

    def gen_school_network(self, age_group, age_fraction):
        self.rescale_schools()
        self.distribute_age_school(age_group, age_fraction)
        self.select_students()
        self.gen_school_position()
        self.alloc_students()

class Workplace:
    def __init__(self, scale, pop_matrix, pop_ages, bl_pop, pop_num, eap_num, eap_ages, eap_ratio_ages, tam_min, tam_max, z3_a, z3_c, a_dist, c_dist):
        self.scale = scale

        self.pop_matrix = pop_matrix
        self.pop_ages = pop_ages
        self.bl_pop = bl_pop
        self.tam_min = tam_min
        self.tam_max = tam_max
        self.z3_a = z3_a
        self.z3_c = z3_c
        self.a_dist = a_dist
        self.c_dist = c_dist

        self.emp_pop_z3 = []
        self.emp_tam_z3 = []
        self.emp_num_z3 = []

        self.emp_bloco_pos = []
        self.emp_por_bloco = []
        self.emp_tam = []

        self.emp_membros_blocos = []
        self.emp_membros = []

        self.pop_num = pop_num
        self.eap_num = eap_num

        self.eap_fractions = []
        self.eap_ages = eap_ages
        self.eap_ratio_ages = eap_ratio_ages

    def zipf3_acum(self, a, c, k_max, k):
        return (((1.0 + k_max/a)/(1.0 + k/a))**c - 1)/((1 + k_max/a)**c - 1.0)

    def zipf3(self, a, c, k_max, k):
        return self.zipf3_acum(a, c, k_max, k-1) - self.zipf3_acum(a, c, k_max, k)


    def zipf3e(self, a, c, k_max, k):
        return self.zipf3(a, c, k_max, k)/k

    def quantifica_empresas_por_tamanho(self, verbose=False):

        emp_tam_z3 = np.arange(self.tam_min, 2*self.tam_max)
        emp_num_z3 = (self.eap_num*self.zipf3e(self.z3_a, self.z3_c, 2*self.tam_max, emp_tam_z3)).astype(int)
        emp_num_z3 = emp_num_z3[emp_num_z3>0]
        emp_tam_z3 = np.array(list(range(self.tam_min, self.tam_min + len(emp_num_z3))))
        emp_pop_z3 = np.array([(self.tam_min + k)*emp_num_z3[k] for k in range(len(emp_num_z3))])

        if not len(emp_tam_z3):
            print('Não foi possível distribuir as empresas, tente com outros parâmetros')
        elif verbose:
            print(f'Total da população: {self.pop_num}')
            print(f'Total da força de trabalho (PEA): {self.eap_num}')
            print(f'Número de tamanhos de empresas: {len(emp_num_z3)}')
            print(f'Número de empresas: {emp_num_z3.sum()}')
            print(f'Tamanhos de empresas: de {emp_tam_z3.min()} a {emp_tam_z3.max()}')
            print(f'Número de indivíduos nas empresas (PEA ocupados): {emp_pop_z3.sum()}')
            print(f'Média de indivíduos por empresa: {emp_pop_z3.sum()/emp_num_z3.sum()}')
            print('Porcentagem de indivíduos da força de trabalho nas empresas: '
                + f'{100*emp_pop_z3.sum()/self.eap_num:.1f}%')
            print(f'Distribuição do número de empresas por tamanho: \n{emp_num_z3}')
            print(f'Distribuição do número de indivíduos por tamanho de empresa: \n{emp_pop_z3}')
        
        self.emp_tam_z3 = emp_tam_z3
        self.emp_num_z3 = emp_num_z3
        self.emp_pop_z3 = emp_pop_z3

    def aloca_empresas(self):

        pop_por_bloco_flat = self.pop_matrix.flatten()
        emp_loc = random.choices(list(range(len(pop_por_bloco_flat))),
                                pop_por_bloco_flat, k=self.emp_num_z3.sum())

        emp_por_bloco = np.zeros_like(self.pop_matrix)

        emp_bloco_pos = list()
        emp_tam = list()
        k_nivel = 0
        for k in range(len(emp_loc)):
            if k >= self.emp_num_z3[:k_nivel+1].sum():
                k_nivel += 1
            emp_tam.append(self.tam_min + k_nivel)
            loc = emp_loc[k]
            emp_bloco_pos.append((loc // 83, loc % 83))
            emp_por_bloco[loc // 83, loc % 83] += 1

        self.emp_bloco_pos = emp_bloco_pos
        self.emp_por_bloco = emp_por_bloco
        self.emp_tam = emp_tam

    def aloca_emp_membros_blocos(self):

        i = np.arange(0.5, 0.5 + self.emp_por_bloco.shape[0])
        j = np.arange(0.5, 0.5 + self.emp_por_bloco.shape[1])
        jj, ii = np.meshgrid(j,i)

        emp_membros_blocos = list()

        f_dist = lambda dist: 1/(1 + (dist/self.a_dist)**self.c_dist)

        for k in range(len(self.emp_num_z3)):
            for j in range(self.emp_num_z3[k]):
                dist =  np.sqrt((jj - self.emp_bloco_pos[k+j][1])**2 
                                + (ii - self.emp_bloco_pos[k+j][0])**2)
                k_dist = f_dist(dist)*self.pop_matrix
                emp_membros_blocos.append(
                    random.choices(
                        list(range(self.emp_por_bloco.shape[0]*self.emp_por_bloco.shape[1])),
                        k_dist.flatten(),
                        k = self.tam_min + k
                    )
                )

        self.emp_membros_blocos = emp_membros_blocos

    def aloca_emp_individuos(self):
        '''
        Aloca os indivíduos em cada empresa.
        '''
        
        indices = np.arange(len(self.pop_ages))
        pop_pia_indices = indices[self.pop_ages >= 16]    
        
        # Define os pesos de cada individuo segundo a sua idade e os pesos para cada idade
        pesos = self.eap_fractions[self.pop_ages[pop_pia_indices]]
        pesos /= pesos.sum() # probabities must add up to 1
        pop_pia_livres = np.random.choice(pop_pia_indices,
                                        size=self.emp_pop_z3.sum(),
                                        replace=False,
                                        p=pesos)
        
        # Escolhe aleatoriamete um indivíduo em cada bloco alocado
        emp_membros = list()

        for j in range(len(self.emp_tam)):
            membros_j = list()
            for l in self.emp_membros_blocos[j]:
                aux = pop_pia_livres[pop_pia_livres >= self.bl_pop[l]]
                candidatos = aux[aux < self.bl_pop[l+1]]
                if len(candidatos):
                    individuo = random.choice(candidatos)
                    membros_j.append(individuo)
                    pop_pia_livres = pop_pia_livres[pop_pia_livres != individuo]
            emp_membros.append(membros_j)

        # Alguns blocos podem não ter mais indivíduos economicamente ativos disponíveis
        # então completamos com indivíduos de outros blocos quaisquer, 
        # portanto, sem peso segundo a distância.
        for j in range(len(self.emp_tam)):
            faltam = self.emp_tam[j] - len(emp_membros[j])
            if faltam > 0:
                membros_j = list(np.random.choice(pop_pia_livres, size=faltam,
                                                replace=False))
                emp_membros[j] += membros_j
                for individuo in membros_j:
                    pop_pia_livres = pop_pia_livres[pop_pia_livres != individuo]
        
        self.emp_membros = emp_membros

    def gen_workplace_network(self, verbose = False):
        self.eap_fractions = episiming.scenes.functions.get_age_fractions(self.eap_ages, self.eap_ratio_ages, age_max=110)
        self.quantifica_empresas_por_tamanho(verbose = verbose)
        self.aloca_empresas()
        self.aloca_emp_membros_blocos()
        self.aloca_emp_individuos()
