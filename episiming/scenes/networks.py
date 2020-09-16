import numpy as np
import random

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