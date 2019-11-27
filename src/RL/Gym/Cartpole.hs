{-# LANGUAGE FlexibleInstances, GeneralizedNewtypeDeriving, MultiParamTypeClasses, TypeFamilies, TypeSynonymInstances #-}
module RL.Gym.Cartpole (Cartpole) where

import Control.Monad.IO.Class
import Control.Monad.Trans.Class
import Control.Monad.Trans.State.Strict
import Data.Maybe
import System.Random
import System.Random.Mersenne.Pure64

import RL.Gym.Types

data CartpoleS = CartpoleS { x :: Double, xdot :: Double, theta :: Double, thetadot :: Double, done :: Bool, gen :: Maybe PureMT }

newtype Cartpole m a = Cartpole { unCartpole :: StateT CartpoleS m a }
  deriving (Functor, Applicative, Monad, MonadTrans)

instance MonadIO m => MonadIO (Cartpole m) where
    liftIO = lift . liftIO

gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = masspole + masscart
lgth = 0.5
polemass_length = masspole * lgth
force_mag = 10
tau = 0.02

theta_threshold_radians = 12 * 2 * pi / 360
x_threshold = 2.4

-- Given an action and a state, compute the new state
advance :: Action Cartpole -> CartpoleS -> CartpoleS
advance a (CartpoleS xx xxd tt ttd d g) = CartpoleS nxx nxxd ntt nttd d g
    where force = case a of
            CLeft -> -force_mag
            CRight -> force_mag
          -- Compute accelerations
          temp = force + polemass_length * ttd * ttd * (sin tt)
          thetaacc = (gravity * (sin tt) - (cos tt) * temp) / (lgth * (4.0 / 3.0 - masspole * (cos tt) * (cos tt) / total_mass))
          xacc = temp - polemass_length * thetaacc * (cos tt) / total_mass
          -- Euler Integrate
          nxx = xx + tau * xxd
          nxxd = xxd + tau * xacc
          ntt = tt + tau * ttd
          nttd = ttd + tau * thetaacc

instance MonadIO m => Gym Cartpole m where
    data Action Cartpole = CLeft | CRight
    data Observation Cartpole = CObs { position :: Double -- [-4.8,4.8]
                                     , velocity :: Double -- (-Inf, Inf)
                                     , angle :: Double -- [-24, 24] degrees
                                     , velocityAtTip :: Double -- (-Inf, Inf)
                                     }
    step a = do
        Cartpole . modify $ advance a
        st <- Cartpole get
        let dn = (abs . x $ st) < x_threshold || (abs . theta $ st) < theta_threshold_radians
            reward = if not dn || not (done st) then 1 else 0
        Cartpole . modify $ \s -> s{done = dn}
        pure $ (CObs (x st) (xdot st) (theta st) (thetadot st), reward, dn)

    run = flip evalStateT (CartpoleS 0 0 0 0 False Nothing) . unCartpole . (seed Nothing >>)
    seed s = do
        g <- case s of
               Just v -> pure (pureMT v)
               Nothing -> liftIO newPureMT
        Cartpole . modify $ \s -> s{gen = Just g}
    reset = do
        g <- fmap (fromJust . gen) (Cartpole get)
        let (p1,g1) = randomR (-0.05,0.05) g
            (p2,g2) = randomR (-0.05,0.05) g1
            (p3,g3) = randomR (-0.05,0.05) g2
            (p4,g4) = randomR (-0.05,0.05) g3
        Cartpole . put $ CartpoleS p1 p2 p3 p4 False (Just g4)
