#ifndef ABC
#define ABC
#include <vector>
#include <random>
#include <algorithm>
#include <boost/numeric/ublas/vector.hpp>

namespace ublas = boost::numeric::ublas;

namespace ABC
{
   template<typename T>
   using Func=std::function<T(ublas::vector<T>)>;
   std::random_device seed_gen;
   std::mt19937 engine(seed_gen());
   std::uniform_real_distribution<double> rand_double(-1, 1);
   template<typename T>
   std::uniform_real_distribution<T> rand_init(-1, 1);

   template<typename T>
   void init(ublas::vector<T> &x)
   {
      for (int i = 0; i < x.size(); i++)
      {
         x[i] = rand_init<T>(engine);
      }
   }
   template<typename T>
   bool update(int const i, std::vector<ublas::vector<T>> &x, std::vector<int> &cnt, ublas::vector<double> &v, Func<T> f)
   {
      int dimension = x[i].size();
      int population = x.size();
      std::uniform_int_distribution<int> rand_dim(0, dimension-1);
      std::uniform_int_distribution<int> rand_pop(0, population-1);
      int j = rand_dim(engine);
      int k = rand_pop(engine);
      double phi = rand_double(engine);
      ublas::vector<T> x_i = ublas::vector<T>(x[i]);
      x_i[j] -= phi*(x_i[j] - x[k][j]);
      double v_new = f(x_i);
      cnt[i]++;
      if(v_new <= v[i])
      {
         x[i] = x_i;
         v[i] = v_new;
         return true;
      }
      return false;
   }
   template<typename T>
   void update_random(int const max_visit, std::vector<ublas::vector<T>> &x, std::vector<int> &cnt, ublas::vector<double> &v, Func<T> f)
   {
      int dimension = x[0].size();
      int population = x.size();
      for(int i = 0; i < population; i++)
      {
         if(cnt[i] < max_visit) continue;
         ublas::vector<T> x_i(dimension);
         init(x_i);
         double v_new = f(x_i);
         cnt[i] = 1;
         if(v_new <= v[i])
         {
            x[i] = x_i;
            v[i] = v_new;
         }
      }
   }
   template<typename T>
   double minimize(int const dimension, int const num_population, int const max_visit, int const max_step, Func<T> f)
   {
      std::vector<ublas::vector<T>> x(num_population);
      std::vector<int> intervals(num_population);
      std::vector<int> cnt(num_population);
      ublas::vector<double> one(num_population);
      ublas::vector<double> v(num_population);
      ublas::vector<double> p(num_population);
      std::uniform_int_distribution<int> rand_dim(0, dimension-1);
      double best_obj = 1 << 20;
      ublas::vector<T> best_x(dimension);      

      for (int i = 0; i < num_population; i++)
      {
         x[i].resize(dimension);
         init(x[i]);
         v[i] = f(x[i]);
         intervals[i] = i;
         one[i] = 1.0;
      }
      for(int step = 0; step < max_step; step++)
      {
         int i = rand_dim(engine);
         update<T>(i, x, cnt, v, f);
         auto m = std::min_element(v.begin(), v.end());
         if(*m < 0)
         {
            ublas::vector<double> w = v - ublas::vector<double>(num_population, *m);
            p = w / ublas::sum(w);
         }
         else
         {
            p = v / ublas::sum(v);
         }
         p = one - p;
         p /= ublas::sum(p);
         std::piecewise_constant_distribution<> dist(intervals.begin(), intervals.end(), p.begin());
         i = static_cast<int>(dist(engine));
         update<T>(i, x, cnt, v, f);
         update_random<T>(max_visit, x, cnt, v, f);

         auto iter =  std::min_element(v.begin(), v.end());
         int idx = std::distance(v.begin(), iter);
         if(best_obj >= *iter)
         {
            best_obj = *iter;
            best_x = x[idx];
         }
      }
      return best_obj;
   }
   
};
#endif // ABC