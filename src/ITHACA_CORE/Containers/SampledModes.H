/*---------------------------------------------------------------------------*\
  ITHACA-FV
  Cached modes on a sampled fvMeshSubset for hyper-reduced projection.
\*---------------------------------------------------------------------------*/

#ifndef SampledModes_H
#define SampledModes_H

#include "fvCFD.H"
#include "fvMeshSubset.H"
#include "SampledMesh.H"

namespace Foam
{

template<class Type, template<class> class PatchField, class GeoMesh>
class SampledModes
{
public:

    typedef GeometricField<Type, PatchField, GeoMesh> fieldType;

private:

    const fvMeshSubset& subset_;

    labelList sampledCells_;
    labelList sampledSubCells_;

    PtrList<fieldType> subModes_;

    void cacheModes
    (
        const PtrList<fieldType>& modes
    )
    {
        if (subModes_.size() > modes.size())
        {
            FatalErrorInFunction
                << "Number of required modes (" << subModes_.size()
                << ") is larger than number of available modes ("
                << modes.size() << ")"
                << exit(FatalError);
        }

        // Expensive mapping: exactly once per mode/subset cache.
        for (label modeI = 0; modeI < subModes_.size(); ++modeI)
        {
            tmp<fieldType> tModeSub =
                subset_.interpolate(modes[modeI]);

            subModes_.set
            (
                modeI,
                new fieldType(tModeSub)
            );
        }
    }

public:

    // Preferred constructor: stencil construction and mapping are already
    // centralized in SampledMesh.
    SampledModes
    (
        const PtrList<fieldType>& modes,
        const SampledMesh& sampledMesh,
        label numberOfModes = 0
    )
    :
        subset_(sampledMesh.subset()),
        sampledCells_(sampledMesh.sampledCells()),
        sampledSubCells_(sampledMesh.sampledSubCells()),
        subModes_
        (
            numberOfModes == 0
          ? modes.size()
          : numberOfModes
        )
    {
        cacheModes(modes);
    }


    // Compatibility constructor for existing callers that already own an
    // fvMeshSubset. Mapping still uses the shared SampledMesh utility.
    SampledModes
    (
        const PtrList<fieldType>& modes,
        const fvMeshSubset& subset,
        const labelList& sampledCells,
        label numberOfModes = 0
    )
    :
        subset_(subset),
        sampledCells_(sampledCells),
        sampledSubCells_
        (
            SampledMesh::mapToSubmesh
            (
                sampledCells,
                subset,
                true
            )
        ),
        subModes_
        (
            numberOfModes == 0
          ? modes.size()
          : numberOfModes
        )
    {
        cacheModes(modes);
    }


    const fvMeshSubset& subset() const
    {
        return subset_;
    }

    const labelList& sampledCells() const
    {
        return sampledCells_;
    }

    const labelList& sampledSubCells() const
    {
        return sampledSubCells_;
    }

    const PtrList<fieldType>& subModes() const
    {
        return subModes_;
    }

    label size() const
    {
        return subModes_.size();
    }

    const fieldType& operator[](const label modeI) const
    {
        return subModes_[modeI];
    }
};

} // End namespace Foam

#endif
