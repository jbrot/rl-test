{-# LANGUAGE MultiParamTypeClasses, TypeFamilies #-}
module RL.Gym.Types where

import Data.Word

class Gym t m where
    data Action t :: *
    data Observation t :: *
    step :: Action t -> t m (Observation t, Float, Bool)
    reset :: t m ()
    seed :: Maybe Word64 -> t m ()
    run :: t m a -> m a
